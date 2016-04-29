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



/********************************** PIConGPU **********************************/


#define HDINLINE inline
#define DIM1 1


#include <boost/type_traits.hpp>

namespace traits {

    template<typename T>
    struct GetValueType
    {
        typedef typename T::ValueType ValueType;
    };

    template<typename Type>
    struct GetValueType<Type*>
    {
        typedef Type ValueType;
    };

}

namespace mappings
{
namespace elements
{


    struct Contiguous
    {
        template< typename T_IdxType >
        HDINLINE T_IdxType
        operator()( const T_IdxType& idx) const
        {
            return idx + 1;
        }
    };


    template< uint32_t T_dim, typename T_Functor, typename T_Size, typename T_Traverse, typename T_Sfinae = void  >
    struct Vectorize;

    template< typename T_Functor, typename T_Size, typename T_Traverse >
    struct Vectorize<
        DIM1,
        T_Functor,
        T_Size,
        T_Traverse,
        typename std::enable_if<
            std::is_integral<
                T_Size
            >::value
        >::type

    >
    {
        inline void
        operator()( const T_Functor& functor, const T_Size& size, const T_Traverse& traverse ) const
        {
            using T_IdxType = T_Size;
            for( T_IdxType i = 0; i < size; ++i)
                functor( i );
        }
    };

    template< uint32_t T_dim, typename T_Functor, typename T_Size, typename T_Traverse = Contiguous>
    HDINLINE void
    vectorize( const T_Functor& functor, const T_Size& size, const T_Traverse& traverse = T_Traverse())
    {
        Vectorize< T_dim, T_Functor, T_Size, T_Traverse >()( functor, size, traverse);
    }

} // namespace elements
} // namespace mappings



template< typename Type >
struct reduce
{
    template< typename Src, typename Dest, class Functor, class Functor2, typename T_Acc>
    ALPAKA_FN_ACC void operator()(
                           const T_Acc& acc,
                           Src src, const uint32_t src_count,
                           Dest dest,
                           Functor func, Functor2 func2) const
    {
        const uint32_t g_localId = threadIdx.x * elemDim.x;
        const uint32_t g_tid = blockIdx.x * (blockDim.x * elemDim.x ) + g_localId;
        const uint32_t globalThreadCount = gridDim.x * blockDim.x * elemDim.x;

        /* cuda can not handle extern shared memory were the type is
         * defined by a template
         * - therefore we use type int for the definition (dirty but OK) */
        sharedMemExtern(s_mem,Type);
        /* create a pointer with the right type*/
        //Type* s_mem=(Type*)s_mem_extern;

        namespace mapElem = mappings::elements;

        mapElem::vectorize<DIM1>(
            [&]( const int idx )
            {
                const uint32_t tid = g_tid + idx;
                const uint32_t localId = g_localId + idx;

                bool isActive = (tid < src_count);

                if(isActive)
                {
                    /*fill shared mem*/
                    Type r_value = src[tid];
                    /*reduce not read global memory to shared*/
                    uint32_t i = tid + globalThreadCount;
                    while (i < src_count)
                    {
                        func(r_value, src[i]);
                        i += globalThreadCount;
                    }
                    s_mem[localId] = r_value;
                }
            },
            elemDim.x,
            mapElem::Contiguous()
        );
        __syncthreads();
        /*now reduce shared memory*/
        uint32_t chunk_count = blockDim.x * elemDim.x;

        while (chunk_count != 1)
        {

            /* Half number of chunks (rounded down) */
            uint32_t active_threads = chunk_count / 2;

            /* New chunks is half number of chunks rounded up for uneven counts
             * --> local_tid=0 will reduce the single element for an odd number of values at the end */
            chunk_count = (chunk_count + 1) / 2;
            mapElem::vectorize<DIM1>(
                [&]( const int idx )
                {
                    const uint32_t tid = g_tid + idx;
                    const uint32_t localId = g_localId + idx;

                    bool isActive = (tid < src_count);
                    isActive = isActive && !(localId != 0 && localId >= active_threads);
                    if(isActive)
                        func(s_mem[localId], s_mem[localId + chunk_count]);
                },
                elemDim.x,
                mapElem::Contiguous()
            );

            __syncthreads();
        }
        if (g_localId==0)
            func2(dest[blockIdx.x], s_mem[0]);
    }
};

class Reduce
{

    struct Assign
    {
        template< typename Dst, typename Src >
                HDINLINE void operator()(Dst & dst, const Src & src) const
        {
            dst = src;
        }
    };


public:

    /* Constructor
     * Don't create a instance before you have set you cuda device!
     * @param byte how many bytes in global gpu memory can reserved for the reduce algorithm
     * @param sharedMemByte limit the usage of shared memory per block on gpu
     */
    inline Reduce(const uint32_t byte, const uint32_t sharedMemByte = 4 * 1024) :
    byte(byte), sharedMemByte(sharedMemByte), reduceBuffer(NULL)
    {

        reduceBuffer = new GridBuffer<char, DIM1 > (DataSpace<DIM1 > (byte));
    }

    /* Reduce elements in global gpu memory
     *
     * @param func binary functor for reduce which takes two arguments, first argument is the source and get the new reduced value.
     * Functor must specialize the function getMPI_Op.
     * @param src a class or a pointer where the reduce algorithm can access the value by operator [] (one dimension access)
     * @param n number of elements to reduce
     *
     * @return reduced value
     */
    template<class Functor, typename Src>
    inline typename traits::GetValueType<Src>::ValueType operator()(Functor func, Src src, uint32_t n)
    {
       /* - the result of a functor can be a reference or a const value
        * - it is not allowed to create const or reference memory
        *   thus we remove `references` and `const` qualifiers */
       typedef typename boost::remove_const<
                   typename boost::remove_reference<
                       typename traits::GetValueType<Src>::ValueType
                   >::type
               >::type Type;

        uint32_t blockcount = optimalThreadsPerBlock(n, sizeof (Type));

        uint32_t n_buffer = byte / sizeof (Type);

        uint32_t threads = n_buffer * blockcount * 2; /* x2 is used thus we can use all byte in Buffer, after we calculate threads/2 */



        if (threads > n) threads = n;
        Type* dest = (Type*) reduceBuffer->getDeviceBuffer().getBasePointer();

        uint32_t blocks = threads / 2 / blockcount;
        if (blocks == 0) blocks = 1;
        __cudaKernel_OPTI(kernel::reduce< Type >)(blocks, blockcount, blockcount * sizeof (Type))(src, n, dest, func,Assign());
        n = blocks;
        blockcount = optimalThreadsPerBlock(n, sizeof (Type));
        blocks = n / 2 / blockcount;
        if (blocks == 0 && n > 1) blocks = 1;


        while (blocks != 0)
        {
            if (blocks > 1)
            {
                uint32_t blockOffset = ceil((double) blocks / blockcount);
                uint32_t useBlocks = blocks - blockOffset;
                uint32_t problemSize = n - (blockOffset * blockcount);
                Type* srcPtr = dest + (blockOffset * blockcount);

                __cudaKernel_OPTI(kernel::reduce< Type >)(useBlocks, blockcount, blockcount * sizeof (Type))(srcPtr, problemSize, dest, func, func);
                blocks = blockOffset*blockcount;
            }
            else
            {

                __cudaKernel_OPTI(kernel::reduce< Type >)(blocks, blockcount, blockcount * sizeof (Type))(dest, n, dest, func,Assign());
            }

            n = blocks;
            blockcount = optimalThreadsPerBlock(n, sizeof (Type));
            blocks = n / 2 / blockcount;
            if (blocks == 0 && n > 1) blocks = 1;
        }

        reduceBuffer->deviceToHost();
        __getTransactionEvent().waitForFinished();
        return *((Type*) (reduceBuffer->getHostBuffer().getBasePointer()));

    }

    virtual ~Reduce()
    {
        __delete(reduceBuffer);
    }

private:

    /* calculate number of threads per block
     * @param threads maximal number of threads per block
     * @return number of threads per block
     */
    inline uint32_t getThreadsPerBlock(uint32_t threads)
    {
        /// \todo this list is not complete
        ///        extend it and maybe check for sm_version
        ///        and add possible threads accordingly.
        ///        maybe this function should be exported
        ///        to a more general nvidia class, too.
        if (threads >= 512) return 512;
        if (threads >= 256) return 256;
        if (threads >= 128) return 128;
        if (threads >= 64) return 64;
        if (threads >= 32) return 32;
        if (threads >= 16) return 16;
        if (threads >= 8) return 8;
        if (threads >= 4) return 4;
        if (threads >= 2) return 2;

        return 1;
    }

    /*calculate optimal number of threads per block with respect to shared memory limitations
     * @param n number of elements to reduce
     * @param sizePerElement size in bytes per elements
     * @return optimal count of threads per block to solve the problem
     */
    inline uint32_t optimalThreadsPerBlock(uint32_t n, uint32_t sizePerElement)
    {
        uint32_t const sharedBorder = sharedMemByte / sizePerElement;
        return getThreadsPerBlock(std::min(sharedBorder, n));
    }

    /*global gpu buffer for reduce steps*/
    GridBuffer<char, DIM1 > *reduceBuffer;
    /*buffer size limit in bytes on gpu*/
    uint32_t byte;
    /*shared memory limit in byte for one block*/
    uint32_t sharedMemByte;

};
