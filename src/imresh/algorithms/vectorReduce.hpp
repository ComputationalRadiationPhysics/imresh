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


#pragma once


#include <cassert>
#include <cstdio>
#include <cstdlib>      // exit
#include <cstdint>      // uint64_t
#include <limits>       // numeric_limits
#include <cuda_to_cupla.hpp>     // atomicCAS, atomicAdd



#define CUDA_ERROR(X) checkCudaError(X,__FILE__,__LINE__);

void checkCudaError
( const cudaError_t rValue, const char * file, int line )
{
    if ( (rValue) != cudaSuccess )
    {
        printf( "CUDA error in %s line:%i : %s\n",
                file, line, cudaGetErrorString(rValue) );
        exit( EXIT_FAILURE );
    }
}

template< typename T >
inline void mallocCudaArray( T ** const rPtr, unsigned int const rnElements )
{
    assert( rnElements > 0 );
    CUDA_ERROR( cudaMalloc( (void**) rPtr, sizeof(T) * rnElements ) );
    assert( rPtr != NULL );
}

template< typename T >
inline void mallocPinnedArray( T ** const rPtr, unsigned int const rnElements )
{
    assert( rnElements > 0 );
    CUDA_ERROR( cudaMallocHost( (void**) rPtr, sizeof(T) * rnElements ) );
    assert( rPtr != NULL );
}


struct CudaKernelConfig
{
    dim3          nBlocks;
    dim3          nThreads;
    int           nBytesSharedMem;
    cudaStream_t  iStream;

    /**
     *
     * Note thate CudaKernelConfig( 1,1,0, cudaStream_t(0) ) is still
     * possible, because dim3( int ) is declared.
     *
     * Blocks and threads can be set to -1 to choose a fitting size
     * automatically. Note that nBytesSharedMem can't be -1, because there
     * is no way to find out automatically what amount is needed.
     */
    CudaKernelConfig
    (
        dim3         rnBlocks         = dim3{ 0, 0, 0 },
        dim3         rnThreads        = dim3{ 0, 0, 0 },
        int          rnBytesSharedMem = 0              ,
        cudaStream_t riStream         = cudaStream_t(0)
    )
    :
        nBlocks        ( rnBlocks         ),
        nThreads       ( rnThreads        ),
        nBytesSharedMem( rnBytesSharedMem ),
        iStream        ( riStream         )
    {
        check();
    }

    /**
     * Checks configuration and autoadjusts default or faulty parameters
     *
     * @return 0: everything was OK, 1: some parameters had to be changed
     */
    int check( void )
    {
        int changed = 0;
        int nMaxBlocks, nMaxThreads;

        #if defined( ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED    ) || \
            defined( ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED ) || \
            defined( ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED     )
            nMaxBlocks  = 1; // number of cores?
            nMaxThreads = 4;
        #endif
        #if defined( ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED )
            nMaxBlocks  = 4; // number of cores?
            nMaxThreads = 1;
        #endif
        #if defined( ALPAKA_ACC_GPU_CUDA_ENABLED )
            /* E.g. GTX 760 can only handle 12288 runnin concurrently,
             * everything else will be run after some threads finished. The
             * number of CUDA cores is only 1152, but they are oversubscribed */
            nMaxThreads = 128;
            nMaxBlocks  = 96;
            //nMaxBlocks  = 192;    // MaxConcurrentThreads / nMaxThreads
        #endif

        if ( nBlocks.x <= 0 )
        {
            changed += 1;
            nBlocks.x = nMaxBlocks;
        }
        if ( nBlocks.y <= 0 )
        {
            changed += 1;
            nBlocks.y = 1;
        }
        if ( nBlocks.z <= 0 )
        {
            changed += 1;
            nBlocks.z = 1;
        }

        if ( nThreads.x <= 0 )
        {
            changed += 1;
            nThreads.x = nMaxThreads;
        }
        if ( nThreads.y <= 0 )
        {
            changed += 1;
            nThreads.y = 1;
        }
        if ( nThreads.z <= 0 )
        {
            changed += 1;
            nThreads.z = 1;
        }

        assert( nBytesSharedMem >= 0 );

        return changed;
    }
};






template<class T_PREC>
T_PREC vectorMax
(
    T_PREC const * const rData,
    unsigned int const rnData,
    unsigned int const rnStride = 1
)
{
    assert( rnStride > 0 );
    T_PREC maximum = std::numeric_limits<T_PREC>::lowest();
    #pragma omp parallel for reduction( max : maximum )
    for ( unsigned i = 0; i < rnData*rnStride; i += rnStride )
        maximum = std::max( maximum, rData[i] );
    return maximum;
}




template< class T_ACC, class T_FUNC >
ALPAKA_FN_ACC_CUDA_ONLY inline void atomicFunc
(
    T_ACC const & acc,
    float * rdpTarget,
    float rValue,
    T_FUNC f
);
/**
 * simple functors to just get the sum of two numbers. To be used
 * for the binary vectorReduce function to make it a vectorSum or
 * vectorMin or vectorMax
 **/
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


template<typename T_ACC>
ALPAKA_FN_ACC_CUDA_ONLY inline void atomicFunc
(
    T_ACC const & acc,
    int * const rdpTarget,
    int   const rValue,
    MaxFunctor<int> f
)
{
    atomicMax( rdpTarget, rValue );
}



#define KERNEL_HEADER(NAME)               \
template< class T_PREC, class T_FUNC >    \
struct NAME                               \
{                                         \
    template< class T_ACC >               \
    ALPAKA_FN_ACC                         \
    void operator()                       \
    (                                     \
        T_ACC const & acc,                \
        T_PREC const * const rdpData,     \
        unsigned int const rnData,        \
        T_PREC * const rdpResult,         \
        T_FUNC f,                         \
        T_PREC const rInitValue           \
    ) const                               \
    {                                     \
        assert( blockDim.y == 1 );        \
        assert( blockDim.z == 1 );        \
        assert( gridDim.y  == 1 );        \
        assert( gridDim.z  == 1 );

KERNEL_HEADER( kernelVectorReduceGlobalAtomic2 )
        const int32_t nTotalThreads = gridDim.x * blockDim.x;
        int32_t i = blockIdx.x * blockDim.x + threadIdx.x;

        #pragma unroll
        for ( ; i < rnData; i += nTotalThreads )
            atomicFunc( acc, rdpResult, rdpData[i], f );
    }
};

KERNEL_HEADER( kernelVectorReduceGlobalAtomic )
        const int32_t nTotalThreads = gridDim.x * blockDim.x;
        int32_t i = blockIdx.x * blockDim.x + threadIdx.x;

        T_PREC localReduced = T_PREC(rInitValue);
        #pragma unroll
        for ( ; i < rnData; i += nTotalThreads )
            localReduced = f( localReduced, rdpData[i] );

        atomicFunc( acc, rdpResult, localReduced, f );
    }
};


KERNEL_HEADER( kernelVectorReduce )
        auto iElem = rdpData + blockIdx.x * blockDim.x + threadIdx.x;
        auto localReduced = T_PREC( rInitValue );
        #pragma unroll
        for ( ; iElem < rdpData + rnData; iElem += gridDim.x * blockDim.x )
            localReduced = f( localReduced, *iElem );

        atomicFunc( acc, rdpResult, localReduced, f );
    }
};

KERNEL_HEADER( kernelVectorReduceSharedMemory )
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
KERNEL_HEADER( kernelVectorReduceSharedMemoryWarps )
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
WARP_REDUCE_WITH_FUNCTOR( SharedMemory )
WARP_REDUCE_WITH_FUNCTOR( SharedMemoryWarps )
WARP_REDUCE_WITH_FUNCTOR( )
#undef WARP_REDUCE_WITH_FUNCTOR
