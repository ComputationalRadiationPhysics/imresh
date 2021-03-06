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
#include <cuda.h>     // atomicCAS, atomicAdd
#include <cufft.h>    // cufftComplex, cufftDoubleComplex
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


    template<class T_PREC, class T_FUNC>
    __device__ inline void atomicFunc
    (
        T_PREC * rdpTarget,
        T_PREC rValue,
        T_FUNC f
    );

    template<class T_FUNC>
    __device__ inline void atomicFunc
    (
        float * rdpTarget,
        float rValue,
        T_FUNC f
    );

    template<class T_FUNC>
    __device__ inline void atomicFunc
    (
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
        __device__ __host__ inline T operator() ( T a, T b )
        { return a+b; }
    };
    template<class T> struct MinFunctor {
        __device__ __host__ inline T operator() ( T a, T b )
        { if (a<b) return a; else return b; } // std::min not possible, can't call host function from device!
    };
    template<class T> struct MaxFunctor {
        __device__ __host__ inline T operator() ( T a, T b )
        { if (a>b) return a; else return b; }
    };
    template<> struct MaxFunctor<float> {
        __device__ __host__ inline float operator() ( float a, float b )
        { return fmax(a,b); }
    };


    template<class T_FUNC>
    __device__ inline void atomicFunc
    (
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
                __float_as_int( f( __int_as_float(assumed), rValue ) ) );
        }
        while ( assumed != old );
    }

    template<class T_FUNC>
    __device__ inline void atomicFunc
    (
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


    template<>
    __device__ inline void atomicFunc<int,MaxFunctor<int>>
    (
        int * const rdpTarget,
        const int rValue,
        MaxFunctor<int> f
    )
    {
        atomicMax( rdpTarget, rValue );
    }


    SumFunctor<float > sumFunctorf;
    MinFunctor<float > minFunctorf;
    MaxFunctor<float > maxFunctorf;
    SumFunctor<double> sumFunctord;
    MinFunctor<double> minFunctord;
    MaxFunctor<double> maxFunctord;


    template<class T_PREC, class T_FUNC>
    __global__ void kernelVectorReduceGlobalAtomic2
    (
        T_PREC const * const rdpData,
        unsigned int const rnData,
        T_PREC * const rdpResult,
        T_FUNC f,
        T_PREC const rInitValue
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        const int32_t nTotalThreads = gridDim.x * blockDim.x;
        int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        assert( i < nTotalThreads );

        for ( ; i < rnData; i += nTotalThreads )
            atomicFunc( rdpResult, rdpData[i], f );
    }


    template<class T_PREC, class T_FUNC>
    __global__ void kernelVectorReduceGlobalAtomic
    (
        T_PREC const * const rdpData,
        unsigned int const rnData,
        T_PREC * const rdpResult,
        T_FUNC f,
        T_PREC const rInitValue
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        const int32_t nTotalThreads = gridDim.x * blockDim.x;
        int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        assert( i < nTotalThreads );

        T_PREC localReduced = T_PREC(rInitValue);
        for ( ; i < rnData; i += nTotalThreads )
            localReduced = f( localReduced, rdpData[i] );

        atomicFunc( rdpResult, localReduced, f );
    }


    template<class T_PREC, class T_FUNC>
    __global__ void kernelVectorReduceSharedMemory
    (
        T_PREC const * const rdpData,
        unsigned int const rnData,
        T_PREC * const rdpResult,
        T_FUNC f,
        T_PREC const rInitValue
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        const int32_t nTotalThreads = gridDim.x * blockDim.x;
        int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        assert( i < nTotalThreads );

        T_PREC localReduced = T_PREC(rInitValue);
        for ( ; i < rnData; i += nTotalThreads )
            localReduced = f( localReduced, rdpData[i] );

        __shared__ T_PREC smReduced;
        /* master thread of every block shall set shared mem variable to 0 */
        __syncthreads();
        if ( threadIdx.x == 0 )
            smReduced = T_PREC(rInitValue);
        __syncthreads();

        atomicFunc( &smReduced, localReduced, f );

        __syncthreads();
        if ( threadIdx.x == 0 )
            atomicFunc( rdpResult, smReduced, f );
    }


    /**
     * benchmarks suggest that this kernel is twice as fast as
     * kernelVectorReduceShared
     **/
    template<class T_PREC, class T_FUNC>
    __global__ void kernelVectorReduceSharedMemoryWarps
    (
        T_PREC const * const rdpData,
        unsigned int const rnData,
        T_PREC * const rdpResult,
        T_FUNC f,
        T_PREC const rInitValue
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

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
        constexpr int warpSize = 32;
        const int32_t laneId = threadIdx.x % warpSize;
        for ( int32_t warpDelta = warpSize / 2; warpDelta > 0; warpDelta /= 2)
            localReduced = f( localReduced, __shfl_down( localReduced, warpDelta ) );

        __shared__ T_PREC smReduced;
        /* master thread of every block shall set shared mem variable to 0 */
        __syncthreads();
        if ( threadIdx.x == 0 )
            smReduced = T_PREC(rInitValue);
        __syncthreads();

        if ( laneId == 0 )
            atomicFunc( &smReduced, localReduced, f );

        __syncthreads();
        if ( threadIdx.x == 0 )
            atomicFunc( rdpResult, smReduced, f );
    }


    #define WRAP_REDUCE_WITH_FUNCTOR( NAME)                                    \
    template<class T_PREC, class T_FUNC>                                       \
    T_PREC cudaReduce##NAME                                                    \
    (                                                                          \
        T_PREC const * const rdpData,                                          \
        unsigned int const rnElements,                                         \
        T_FUNC f,                                                              \
        T_PREC const rInitValue,                                               \
        cudaStream_t rStream                                                   \
    )                                                                          \
    {                                                                          \
        /* the more threads we have the longer the reduction will be           \
         * done inside shared memory instead of global memory */               \
        const unsigned nThreads = 256;                                         \
        const unsigned nBlocks = 256;                                          \
        assert( nBlocks < 65536 );                                             \
                                                                               \
        T_PREC reducedValue;                                                   \
        T_PREC * dpReducedValue;                                               \
        T_PREC initValue = rInitValue;                                         \
                                                                               \
        CUDA_ERROR( cudaMalloc( (void**) &dpReducedValue, sizeof(T_PREC) ) );  \
        CUDA_ERROR( cudaMemcpyAsync( dpReducedValue, &initValue,               \
                                     sizeof(T_PREC),                           \
                                     cudaMemcpyHostToDevice,                   \
                                     rStream ) );                              \
                                                                               \
        /* memcpy is on the same stream as kernel will be, so no synchronize   \
           needed! */                                                          \
        kernelVectorReduce##NAME<<< nBlocks, nThreads, 0, rStream >>>          \
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
        T_PREC const * const rdpData,                                          \
        unsigned int const rnElements,                                         \
        cudaStream_t rStream                                                   \
    )                                                                          \
    {                                                                          \
        MaxFunctor<T_PREC> maxFunctor;                                         \
        return cudaReduce##NAME( rdpData, rnElements, maxFunctor,              \
                                 std::numeric_limits<T_PREC>::lowest(),        \
                                 rStream );                                    \
    }                                                                          \
                                                                               \
    /* explicit template instantiations */                                     \
                                                                               \
    template                                                                   \
    float cudaVectorMax##NAME<float>                                           \
    (                                                                          \
        float const * const rdpData,                                           \
        unsigned int const rnElements,                                         \
        cudaStream_t rStream                                                   \
    );


    WRAP_REDUCE_WITH_FUNCTOR( GlobalAtomic2 )
    WRAP_REDUCE_WITH_FUNCTOR( GlobalAtomic )
    WRAP_REDUCE_WITH_FUNCTOR( SharedMemory )
    WRAP_REDUCE_WITH_FUNCTOR( SharedMemoryWarps )

    inline __device__ uint32_t getLaneId( void )
    {
        uint32_t id;
        asm("mov.u32 %0, %%laneid;" : "=r"(id));
        return id;
    }

    /* needs __CUDA_ARCH__ >= 200 */
    inline __device__ uint32_t bfe
    (
        uint32_t src,
        uint32_t offset,
        uint32_t nBits
    )
    {
        uint32_t result;
        asm( "bfe.u32 %0, %1, %2, %3;" :
             "=r"(result) : "r"(src), "r"(offset), "r"(nBits) );
        return result;
    }

    /**
     * @see cudaKernelCalculateHioError
     **/
    template<class T_COMPLEX>
    __global__ void cudaKernelCalculateHioErrorBitPacked
    (
        T_COMPLEX  const * const __restrict__ rdpData,
        uint32_t   const * const __restrict__ rdpIsMasked,
        unsigned int const rnData,
        float * const __restrict__ rdpTotalError,
        float * const __restrict__ rdpnMaskedPixels
    )
    {
        /**
         * @see http://www.pixel.io/blog/2012/4/19/does-anyone-actually-use-cudas-built-in-warpsize-variable.html
         * warpSize will be read with some assembler isntruction, therefore
         * it is not known at compile time, meaning some optimizations like
         * loop unrolling won't work. That's the reason for this roundabout way
         **/
        constexpr int cWarpSize = 32;
        static_assert( cWarpSize == 8 * sizeof( rdpIsMasked[0] ), "" );
        assert( cWarpSize  == warpSize );
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
            bool const shouldBeZero = bfe(
                rdpIsMasked[ i/nBits ], nBits-1 - threadIdx.x, 1 );

            auto const re = rdpData[i].x;
            auto const im = rdpData[i].y;

            localTotalError    += shouldBeZero * sqrtf( re*re+im*im );
            localnMaskedPixels += shouldBeZero;
        }

        #pragma unroll
        for ( int warpDelta = cWarpSize / 2; warpDelta > 0; warpDelta /= 2 )
        {
            localTotalError    += __shfl_down( localTotalError   , warpDelta );
            localnMaskedPixels += __shfl_down( localnMaskedPixels, warpDelta );
        }

        if ( getLaneId() == 0 )
        {
            atomicAdd( rdpTotalError   , localTotalError    );
            atomicAdd( rdpnMaskedPixels, localnMaskedPixels );
        }
    }


    template<class T_COMPLEX>
    float cudaCalculateHioErrorBitPacked
    (
        T_COMPLEX const * const rdpData,
        uint32_t  const * const rdpIsMasked,
        unsigned int const rnElements,
        bool const rInvertMask,
        cudaStream_t rStream,
        float * const rpTotalError,
        float * const rpnMaskedPixels
    )
    {
        const unsigned nThreads = 32;   /* must be warpSize! */
        //const unsigned nBlocks  = ceil( (float) rnElements / nThreads );
        const unsigned nBlocks  = 256;
        assert( nBlocks < 65536 );

        float     totalError,     nMaskedPixels;
        float * dpTotalError, * dpnMaskedPixels;

        CUDA_ERROR( cudaMalloc( (void**) &dpTotalError   , sizeof(float) ) );
        CUDA_ERROR( cudaMalloc( (void**) &dpnMaskedPixels, sizeof(float) ) );
        CUDA_ERROR( cudaMemsetAsync( dpTotalError   , 0, sizeof(float), rStream ) );
        CUDA_ERROR( cudaMemsetAsync( dpnMaskedPixels, 0, sizeof(float), rStream ) );

        /* memset is on the same stream as kernel will be, so no synchronize needed! */
        cudaKernelCalculateHioErrorBitPacked<<< nBlocks, nThreads, 0, rStream >>>
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



    /* explicit template instantiations */

    template
    __global__ void cudaKernelCalculateHioErrorBitPacked<cufftComplex>
    (
        cufftComplex const * const __restrict__ rdpgPrime,
        uint32_t     const * const __restrict__ rdpIsMasked,
        unsigned int const rnData,
        float * const __restrict__ rdpTotalError,
        float * const __restrict__ rdpnMaskedPixels
    );

    template
    float cudaCalculateHioErrorBitPacked<cufftComplex>
    (
        cufftComplex const * const rdpData,
        uint32_t  const * const rdpIsMasked,
        unsigned int const rnElements,
        bool const rInvertMask,
        cudaStream_t rStream,
        float * const rpTotalError,
        float * const rpnMaskedPixels
    );


} // namespace cuda
} // namespace algorithms
} // namespace imresh
} // namespace benchmark
