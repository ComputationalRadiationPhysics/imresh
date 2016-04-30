/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2016 Maximilian Knespel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "cudaVectorReduce.hpp"

#include <cassert>
#include <cstdio>
#include <cstdint>    // uint64_t
#include <limits>     // numeric_limits
#include <cuda_to_cupla.hpp>     // atomicCAS, atomicAdd
#include "libs/cufft_to_cupla.hpp"    // cufftComplex, cufftDoubleComplex
#include "libs/cudacommon.hpp"
/**
 * Gives only compile errors, e.g.
 *    ptxas fatal   : Unresolved extern function '_ZN6imresh10algorithms4cuda10SumFunctorIfEclEff'
 * so I justd copy-pasted the functors here ...
 **/
//#include "algorithms/cuda/cudaVectorReduce.hpp" // maxFunctor, atomicFunc


namespace benchmark
{
namespace imresh
{
namespace algorithms
{
namespace cuda
{


    template< class T_ACC, class T_PREC, class T_FUNC >
    ALPAKA_FN_ACC_CUDA_ONLY inline void atomicFunc
    (
        T_ACC const & acc,
        T_PREC * rdpTarget,
        T_PREC rValue,
        T_FUNC f
    );

    template< class T_ACC, class T_FUNC >
    ALPAKA_FN_ACC_CUDA_ONLY inline void atomicFunc
    (
        T_ACC const & acc,
        float * rdpTarget,
        float rValue,
        T_FUNC f
    );

    template< class T_ACC, class T_FUNC >
    ALPAKA_FN_ACC_CUDA_ONLY inline void atomicFunc
    (
        T_ACC const & acc,
        double * rdpTarget,
        double rValue,
        T_FUNC f
    );

    /**
     * simple functors to just get the sum of two numbers. To be used
     * for the binary vectorReduce function to make it a vectorSum or
     * vectorMin or vectorMax
     **/
    template<class T> struct SumFunctor {
        ALPAKA_FN_ACC_CUDA_ONLY inline T operator() ( T a, T b )
        { return a+b; }
    };
    template<class T> struct MinFunctor {
        ALPAKA_FN_ACC_CUDA_ONLY inline T operator() ( T a, T b )
        { if (a<b) return a; else return b; } // std::min not possible, can't call host function from device!
    };
    template<class T> struct MaxFunctor {
        ALPAKA_FN_ACC_CUDA_ONLY inline T operator() ( T a, T b )
        { if (a>b) return a; else return b; }
    };
    template<> struct MaxFunctor<float> {
        ALPAKA_FN_ACC_CUDA_ONLY inline float operator() ( float a, float b )
        { return fmax(a,b); }
    };


    template< class T_ACC, class T_FUNC >
    ALPAKA_FN_ACC_CUDA_ONLY inline void atomicFunc
    (
        T_ACC const & acc,
        float * const rdpTarget,
        const float rValue,
        T_FUNC f
    )
    {
        uint32_t assumed;
        uint32_t old = * (uint32_t*) rdpTarget;

        do
        {
            assumed = old;
            old = atomicCAS( (uint32_t*) rdpTarget, assumed,
                (uint32_t) __float_as_int( f( __int_as_float(assumed), rValue ) ) );
        }
        while ( assumed != old );
    }

    template< class T_ACC, class T_FUNC >
    ALPAKA_FN_ACC_CUDA_ONLY inline void atomicFunc
    (
        T_ACC const & acc,
        double * const rdpTarget,
        const double rValue,
        T_FUNC f
    )
    {
        using ull = unsigned long long int;
        ull assumed;
        ull old = * (ull*) rdpTarget;
        do
        {
            assumed = old;
            old = atomicCAS( (ull*) rdpTarget, assumed,
                __double_as_longlong( f( __longlong_as_double(assumed), rValue ) ) );
        }
        while ( assumed != old );
    }

    /*
    template< class T_ACC >
    ALPAKA_FN_ACC_CUDA_ONLY inline
    void atomicFunc< T_ACC, int, MaxFunctor<int> >
    (
        T_ACC const & acc,
        int * const rdpTarget,
        const int rValue,
        MaxFunctor<int> f
    )
    {
        atomicMax( rdpTarget, rValue );
    }
    */

    #define REDUCE_KERNEL_HEADER( NAME )   \
    template< class T_PREC, class T_FUNC > \
    struct NAME                            \
    {                                      \
        template< class T_ACC >            \
        ALPAKA_FN_ACC                      \
        void operator()                    \
        (                                  \
            T_ACC const & acc,             \
            T_PREC const * const rdpData,  \
            unsigned int const rnData,     \
            T_PREC * const rdpResult,      \
            T_FUNC f,                      \
            T_PREC const rInitValue        \
        ) const                            \
        {                                  \
            assert( blockDim.y == 1 );     \
            assert( blockDim.z == 1 );     \
            assert( gridDim.y  == 1 );     \
            assert( gridDim.z  == 1 );

    REDUCE_KERNEL_HEADER( kernelVectorReduceGlobalAtomic2 )
            const int32_t nTotalThreads = gridDim.x * blockDim.x;
            int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            assert( i < nTotalThreads );

            #pragma unroll
            for ( ; i < rnData; i += nTotalThreads )
                atomicFunc( acc, rdpResult, rdpData[i], f );
        }
    };

    REDUCE_KERNEL_HEADER( kernelVectorReduceGlobalAtomic )
            int32_t const nTotalThreads = gridDim.x * blockDim.x;
            int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            assert( i < nTotalThreads );

            auto localReduced = T_PREC( rInitValue );
            /* not having lastReduced and nTotalThreads results in a 2x slowdown! -> see Pointer kernel */
            #pragma unroll
            for ( ; i < rnData; i += nTotalThreads )
                localReduced = f( localReduced, rdpData[i] );

            atomicFunc( acc, rdpResult, localReduced, f );
        }
    };

    REDUCE_KERNEL_HEADER( kernelVectorReducePointer )
            auto iElem = rdpData + blockIdx.x * blockDim.x + threadIdx.x;
            auto localReduced = T_PREC( rInitValue );

            /* not having lastReduced and nTotalThreads results in a 2.5x
             * slowdown! I guess because gridDim and blockDim are non-const
             * when using alpaka -.-
             * Even when using auto instead of auto const copying gridDim.x
             * and blockDim.x to a local variable makes this 2.5x faster!
             * only if used directly it's so slow */
            #define VERSION 2
            #if VERSION == 0
                auto const nBlocks  = gridDim.x;
                auto const nThreads = blockDim.x;
            #elif VERSION == 1
                auto nBlocks  = gridDim.x;
                auto nThreads = blockDim.x;
            #endif
            assert( acc.m_gridBlockExtent[2]   == gridDim.x  );
            assert( acc.m_blockThreadExtent[2] == blockDim.x );
            #pragma unroll
            for ( ; iElem < rdpData + rnData; iElem +=
                        #if VERSION == 2
                            gridDim.x * blockDim.x
                        #elif VERSION == 3
                            acc.m_gridBlockExtent[2] * acc.m_blockThreadExtent[2]
                        #else
                            nBlocks * nThreads
                        #endif
                                                        )
                localReduced = f( localReduced, *iElem );

            atomicFunc( acc, rdpResult, localReduced, f );
            /* average time needed using OpenMP for 16777216 elements:
             * VERSION 0: 29 ms
             * VERSION 1: 29 ms
             * VERSION 2: 76 ms
             * VERSION 3: 29 ms
             */
        }
    };

    REDUCE_KERNEL_HEADER( kernelVectorReduceSharedMemory )
            const int32_t nTotalThreads = gridDim.x * blockDim.x;
            int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            assert( i < nTotalThreads );

            T_PREC localReduced = T_PREC(rInitValue);
            #pragma unroll
            for ( ; i < rnData; i += nTotalThreads )
                localReduced = f( localReduced, rdpData[i] );

            sharedMem( smReduced, T_PREC );
            /* master thread of every block shall set shared mem variable to 0 */
            __syncthreads();
            if ( threadIdx.x == 0 )
                smReduced = T_PREC(rInitValue);
            __syncthreads();

            atomicFunc( acc, &smReduced, localReduced, f );

            __syncthreads();
            if ( threadIdx.x == 0 )
                atomicFunc( acc, rdpResult, smReduced, f );
        }
    };

    /**
     * benchmarks suggest that this kernel is twice as fast as
     * kernelVectorReduceShared
     **/
    REDUCE_KERNEL_HEADER( kernelVectorReduceSharedMemoryWarps )
            const int32_t nTotalThreads = gridDim.x * blockDim.x;
            int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            assert( i < nTotalThreads );

            T_PREC localReduced = T_PREC(rInitValue);
            for ( ; i < rnData; i += nTotalThreads )
                localReduced = f( localReduced, rdpData[i] );

            /**
             * reduce per warp:
             * With __shfl_down we can read the register values of other lanes in
             * a warp. In the first iteration lane 0 will add to it's value the
             * value of lane 16, lane 1 from lane 17 and so in.
             * In the next step lane 0 will add the result from lane 8.
             * In the end lane 0 will have the reduced value.
             * @see http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
             **/
            //constexpr int warpSize = 32;
            //const int32_t laneId = threadIdx.x % warpSize;
            //for ( int32_t warpDelta = warpSize / 2; warpDelta > 0; warpDelta /= 2)
            //    localReduced = f( localReduced, __shfl_down( localReduced, warpDelta ) );

            sharedMem( smReduced, T_PREC );
            /* master thread of every block shall set shared mem variable to 0 */
            __syncthreads();
            if ( threadIdx.x == 0 )
                smReduced = T_PREC(rInitValue);
            __syncthreads();

            //if ( laneId == 0 )
                atomicFunc( acc, &smReduced, localReduced, f );

            __syncthreads();
            if ( threadIdx.x == 0 )
                atomicFunc( acc, rdpResult, smReduced, f );
        }
    };

    #define WARP_REDUCE_WITH_FUNCTOR( NAME)                                    \
    template<class T_PREC, class T_FUNC>                                       \
    T_PREC cudaReduce##NAME                                                    \
    (                                                                          \
        CudaKernelConfig rKernelConfig,                                        \
        T_PREC const * const rdpData,                                          \
        unsigned int const rnElements,                                         \
        T_FUNC f,                                                              \
        T_PREC const rInitValue                                                \
    )                                                                          \
    {                                                                          \
        using ::imresh::libs::mallocCudaArray;                                   \
                                                                               \
        auto const & rStream = rKernelConfig.iStream;                          \
        /* the more threads we have the longer the reduction will be           \
         * done inside shared memory instead of global memory */               \
                                                                               \
        T_PREC reducedValue;                                                   \
        T_PREC * dpReducedValue;                                               \
        T_PREC initValue = rInitValue;                                         \
                                                                               \
        mallocCudaArray( &dpReducedValue, 1 );                                 \
        CUDA_ERROR( cudaMemcpyAsync( dpReducedValue, &initValue,               \
                                     sizeof(T_PREC),                           \
                                     cudaMemcpyHostToDevice,                   \
                                     rStream ) );                              \
                                                                               \
        /* memcpy is on the same stream as kernel will be, so no synchronize   \
           needed! */                                                          \
        CUPLA_KERNEL                                                           \
            ( kernelVectorReduce##NAME< T_PREC, T_FUNC> )                      \
            ( rKernelConfig.nBlocks, rKernelConfig.nThreads, 0, rStream )      \
            ( rdpData, rnElements, dpReducedValue, f, rInitValue );            \
                                                                               \
        CUDA_ERROR( cudaStreamSynchronize( rStream ) );                        \
        CUDA_ERROR( cudaMemcpyAsync( &reducedValue, dpReducedValue,            \
                                     sizeof(T_PREC),                           \
                                     cudaMemcpyDeviceToHost, rStream ) );      \
        CUDA_ERROR( cudaStreamSynchronize( rStream) );                         \
        CUDA_ERROR( cudaFree( dpReducedValue ) );                              \
                                                                               \
        return reducedValue;                                                   \
    }                                                                          \
                                                                               \
    template<class T_PREC>                                                     \
    T_PREC cudaVectorMax##NAME                                                 \
    (                                                                          \
        CudaKernelConfig rKernelConfig,                                        \
        T_PREC const * const rdpData,                                          \
        unsigned int const rnElements                                          \
    )                                                                          \
    {                                                                          \
        MaxFunctor<T_PREC> maxFunctor;                                         \
        return cudaReduce##NAME(                                               \
            rKernelConfig,                                                     \
            rdpData, rnElements, maxFunctor,                                   \
            std::numeric_limits<T_PREC>::lowest()                              \
        );                                                                     \
    }
    WARP_REDUCE_WITH_FUNCTOR( GlobalAtomic2 )
    WARP_REDUCE_WITH_FUNCTOR( GlobalAtomic )
    WARP_REDUCE_WITH_FUNCTOR( Pointer )
    WARP_REDUCE_WITH_FUNCTOR( SharedMemory )
    WARP_REDUCE_WITH_FUNCTOR( SharedMemoryWarps )
    #undef WARP_REDUCE_WITH_FUNCTOR

    inline ALPAKA_FN_ACC_CUDA_ONLY uint32_t getLaneId( void )
    {
        uint32_t id;
        asm("mov.u32 %0, %%laneid;" : "=r"(id));
        return id;
    }

    /* needs __CUDA_ARCH__ >= 200 */
    /**
     * returns nBits starting from offset of src. Bit 0 is the lowest value.
     **/
    inline ALPAKA_FN_ACC uint32_t bfe
    (
        uint32_t src,
        uint32_t offset,
        uint32_t nBits
    )
    {
        uint32_t result;
        #if __CUDACC__
            asm( "bfe.u32 %0, %1, %2, %3;" :
                 "=r"(result) : "r"(src), "r"(offset), "r"(nBits) );
        #else
            result = ( ( uint32_t(0xFFFFFFFF) >> (32-nBits) ) << offset ) & src;
        #endif
        return result;
    }

    /**
     * @see cudaKernelCalculateHioError
     **/
    template<class T_COMPLEX>
    struct cudaKernelCalculateHioErrorBitPacked
    {
        template< typename T_ACC >
        ALPAKA_FN_ACC
        void operator()
        (
            T_ACC             const &            acc             ,
            T_COMPLEX const * const __restrict__ rdpData         ,
            uint32_t  const * const __restrict__ rdpIsMasked     ,
            unsigned int      const              rnData          ,
            float           * const __restrict__ rdpTotalError   ,
            float           * const __restrict__ rdpnMaskedPixels
        ) const
        {
            /**
             * @see http://www.pixel.io/blog/2012/4/19/does-anyone-actually-use-cudas-built-in-warpsize-variable.html
             * warpSize will be read with some assembler instruction, therefore
             * it is not known at compile time, meaning some optimizations like
             * loop unrolling won't work. That's the reason for this roundabout way
             **/
            constexpr int cWarpSize = 32;
            static_assert( cWarpSize == 8 * sizeof( rdpIsMasked[0] ), "" );
            //assert( cWarpSize  == warpSize );
            assert( blockDim.x == cWarpSize );
            assert( blockDim.y == 1 );
            assert( blockDim.z == 1 );
            assert( gridDim.y  == 1 );
            assert( gridDim.z  == 1 );

            uint32_t constexpr nBits = sizeof(rdpIsMasked[0]) * 8;
            int const nTotalThreads = gridDim.x * blockDim.x;
            auto i = blockIdx.x * blockDim.x + threadIdx.x;

            float localTotalError    = 0;
            float localnMaskedPixels = 0;
            #pragma unroll
            for ( ; i < rnData; i += nTotalThreads )
            {
                assert( i % nBits == threadIdx.x );
                bool const shouldBeZero = bfe(
                    rdpIsMasked[ i/nBits ], nBits-1 - threadIdx.x, 1 );

                auto const re = rdpData[i].x;
                auto const im = rdpData[i].y;

                localTotalError    += shouldBeZero * sqrtf( re*re+im*im );
                localnMaskedPixels += shouldBeZero;
            }
            atomicAdd( rdpTotalError   , localTotalError    );
            atomicAdd( rdpnMaskedPixels, localnMaskedPixels );
            __syncthreads();
        }
    };


    template<class T_COMPLEX>
    float cudaCalculateHioErrorBitPacked
    (
        CudaKernelConfig rKernelConfig,
        T_COMPLEX const * const rdpData,
        uint32_t  const * const rdpIsMasked,
        unsigned int const rnElements,
        bool const rInvertMask,
        float * const rpTotalError,
        float * const rpnMaskedPixels
    )
    {
        using ::imresh::libs::mallocCudaArray;

        auto const & rStream = rKernelConfig.iStream;

        float     totalError,     nMaskedPixels;
        float * dpTotalError, * dpnMaskedPixels;

        mallocCudaArray( &dpTotalError   , 1 );
        mallocCudaArray( &dpnMaskedPixels, 1 );
        CUDA_ERROR( cudaMemsetAsync( dpTotalError   , 0, sizeof(float), rStream ) );
        CUDA_ERROR( cudaMemsetAsync( dpnMaskedPixels, 0, sizeof(float), rStream ) );

        /* memset is on the same stream as kernel will be, so no synchronize needed! */
        CUPLA_KERNEL( cudaKernelCalculateHioErrorBitPacked<T_COMPLEX> )
            ( rKernelConfig.nBlocks, rKernelConfig.nThreads, 0, rKernelConfig.iStream )
            ( rdpData, rdpIsMasked, rnElements, dpTotalError, dpnMaskedPixels );
        CUDA_ERROR( cudaStreamSynchronize( rStream ) );

        CUDA_ERROR( cudaMemcpyAsync( &totalError   , dpTotalError   , sizeof(float), cudaMemcpyDeviceToHost, rStream ) );
        CUDA_ERROR( cudaMemcpyAsync( &nMaskedPixels, dpnMaskedPixels, sizeof(float), cudaMemcpyDeviceToHost, rStream ) );
        CUDA_ERROR( cudaStreamSynchronize( rStream ) );

        CUDA_ERROR( cudaFree( dpTotalError    ) );
        CUDA_ERROR( cudaFree( dpnMaskedPixels ) );

        if ( rpTotalError != NULL )
            *rpTotalError    = totalError;
        if ( rpnMaskedPixels != NULL )
            *rpnMaskedPixels = nMaskedPixels;

        return sqrtf(totalError) / nMaskedPixels;
    }


} // namespace cuda
} // namespace algorithms
} // namespace imresh
} // namespace benchmark
