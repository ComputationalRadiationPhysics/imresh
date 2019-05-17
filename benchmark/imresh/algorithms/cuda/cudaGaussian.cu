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


#include "cudaGaussian.hpp"

#include <iostream>
#include <cstdio>           // printf
#include <cstdlib>          // exit, EXIT_FAILURE
#include <cassert>
#include <cstddef>          // NULL
#include <map>
#include <mutex>
#include <list>
#include <utility>          // pair
#include <cuda.h>
#include "libs/calcGaussianKernel.hpp"
#include "libs/cudacommon.hpp"


/* constant memory declaration doesn't work inside namespaces */
constexpr unsigned nMaxWeights = 50; // need to assert whether this is enough
constexpr unsigned nMaxKernels = 20; // ibid
__constant__ float gdpGaussianWeights[ nMaxWeights*nMaxKernels ];


namespace benchmark
{
namespace imresh
{
namespace algorithms
{
namespace cuda
{


    /** this flag adds some debug output to stdout if set to 1 **/
    #define DEBUG_CUDAGAUSSIAN_CPP 0

    template<class T>
    __device__ inline T * ptrMin ( T * const a, T * const b )
    {
        return a < b ? a : b;
    }

    /**
     * Provides a class for a moving window type 2d cache
     **/
    template<class T_PREC>
    struct Cache1d
    {
        T_PREC const * const & data;
        unsigned const & nData;

        T_PREC * const & smBuffer; /**< pointer to allocated buffer, will not be allocated on constructor because this class needs to be trivial to work on GPU */
        unsigned const & nBuffer;

        unsigned const & nThreads;
        unsigned const & nKernelHalf;

        __device__ inline T_PREC & operator[]( unsigned i ) const
        {
            return smBuffer[i];
        }

        #ifndef NDEBUG
        #if DEBUG_CUDAGAUSSIAN_CPP == 1
            __device__ void printCache( void ) const
            {
                if ( threadIdx.x != 0 or blockIdx.x != 0 )
                    return;
                for ( unsigned i = 0; i < nBuffer; ++i )
                {
                    printf( "% 3i :", i );
                    printf( "%11.6f\n", smBuffer[i] );
                }
            }
        #endif
        #endif

        __device__ inline void initializeCache( void ) const
        {
            #ifndef NDEBUG
            #if DEBUG_CUDAGAUSSIAN_CPP == 1
                /* makes it easier to see if we cache the correct data */
                if ( threadIdx.x == 0 )
                    memset( smBuffer, 0, nBuffer*sizeof( smBuffer[0] ) );
                __syncthreads();
            #endif
            #endif

            /* In the first step initialize the left border to the same values (extend)
             * It's problematic to use threads for this for loop, because it is not
             * guaranteed, that blockDim.x >= N */
            /**
             * benchmark ImageSize 1024x1024
             *    parallel    : 1.55ms
             *    non-parallel: 1.59ms
             * the used register count is equal for both versions.
             **/
            #if false
                for ( T_PREC * target = smBuffer + nThreads + threadIdx.x;
                      target < smBuffer + nBuffer; target += nThreads )
                {
                    *target = leftBorderValue;
                }
            #else
                if ( threadIdx.x == 0 )
                for ( unsigned iB = nThreads; iB < nBuffer; ++iB )
                {
                    const int signedDataIndex = int(iB-nThreads) - (int)nKernelHalf;
                    /* periodic */
                    //unsigned cappedIndex = signedDataIndex % signedDataIndex;
                    //if ( cappedIndex < 0 ) cappedIndex += rImageWidth;
                    /* extend */
                    const unsigned cappedIndex = min( nData-1, (unsigned) max( 0, signedDataIndex ) );
                    smBuffer[iB] = data[cappedIndex];
                }
            #endif

            #ifndef NDEBUG
            #if DEBUG_CUDAGAUSSIAN_CPP == 1
                if ( threadIdx.x == 0 and blockIdx.x == 0 )
                {
                    printf("Copy some initial data to buffer:\n");
                    printCache();
                }
            #endif
            #endif
        }

        __device__ inline void loadCacheLine( T_PREC const * const curDataRow ) const
        {
            /* move last N elements to the front of the buffer */
            __syncthreads();
            /**
             * all of these do the same, but benchmarks suggest that the
             * last version which looks the most complicated is the fastest:
             * imageSize 1024x1024, compiled with -O0:
             *   - memcpy            : 1.09ms
             *   - parallel          : 0.78ms
             *   - pointer arithmetic: 0.89ms
             * not that the memcpy version only seems to work on the GPU, on
             * the CPU it lead to wrong results I guess because we partially
             * write to memory we also read, or it was some kind of other
             * error ...
             **/
            #if true
                /* eliminating the variable i doesn't make it faster ... */
                for ( unsigned i = threadIdx.x, i0 = 0;
                      i0 + nThreads < nBuffer;
                      i += nThreads, i0 += nThreads )
                {
                    if ( i+nThreads < nBuffer )
                        smBuffer[i] = smBuffer[i+nThreads];
                    __syncthreads();
                }
            #elif true
                if ( threadIdx.x == 0 )
                    memcpy( smBuffer, smBuffer + nThreads, nKernelHalf*sizeof(T_PREC) );
            #else
                /* this version may actually be wrong, because some threads
                 * could be already in next iteration, thereby overwriting
                 * some values which still are to be moved! */
                /*{
                    T_PREC * iTarget = smBuffer + threadIdx.x;
                    T_PREC const * iSrc = iTarget + nThreads;
                    for ( ; iSrc < smBuffer + nBufferSize;
                          iTarget += nThreads, iSrc += nThreads )
                    {
                        *iTarget = *iSrc;
                    }
                }*/
            #endif

            #ifndef NDEBUG
            #if DEBUG_CUDAGAUSSIAN_CPP == 1
                if ( threadIdx.x == 0 and blockIdx.x == 0 )
                {
                    printf( "Shift buffer by %i elements:\n", nThreads );
                    printCache();
                }
            #endif
            #endif

            /* Load nThreads new data elements into buffer. */
            //for ( unsigned iRowBuf = nRowsBuffer - nThreads; iRowBuf < nRowsBuffer; ++iRowBuf )
            //const unsigned newRow = min( rnDataY-1, (unsigned) max( 0,
            //    (int)iRow - (int)nKernelHalf + (int)iRowBuf ) );
            //const unsigned iBuf = nKernelHalf + threadIdx.x;
            const unsigned iBuf = nBuffer - nThreads + threadIdx.x;
            /* If data end reached, fill buffer with last data element */
            assert( curDataRow - nKernelHalf + iBuf == curDataRow + nKernelHalf + threadIdx.x );
            T_PREC const * const datum = ptrMin( data + nData-1,
                curDataRow - nKernelHalf + iBuf );
            assert( iBuf < nBuffer );
            __syncthreads();
            smBuffer[ iBuf ] = *datum;
            __syncthreads();

            #ifndef NDEBUG
            #if DEBUG_CUDAGAUSSIAN_CPP == 1
                if ( threadIdx.x == 0 and blockIdx.x == 0 )
                {
                    printf( "Load %i new elements:\n", nThreads );
                    printCache();
                }
            #endif
            #endif
        }
    };


    template<class T_PREC>
    __global__ void cudaKernelApplyKernelConstantWeights
    (
        /* You can't pass by reference to a kernel !!! compiles, but gives weird errors ... */
        T_PREC * const rdpData,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        T_PREC * const rWeights,
        unsigned int const rnWeights
    )
    {
        assert( blockDim.y == 1 and blockDim.z == 1 );
        assert(  gridDim.y == 1 and  gridDim.z == 1 );
        assert( rnWeights >= 1 );
        const unsigned nKernelHalf = (rnWeights-1) / 2;

        const int & nThreads = blockDim.x;
        T_PREC * const data = &rdpData[ blockIdx.x * rnDataX ];

        /* @see http://stackoverflow.com/questions/27570552/ */
        extern __shared__ __align__( sizeof(T_PREC) ) unsigned char sm[];
        T_PREC * const smBuffer = reinterpret_cast<T_PREC*>( sm );
        const unsigned nBufferSize = nThreads + 2*nKernelHalf;
        __syncthreads();

        Cache1d<T_PREC> buffer{ data, rnDataX, smBuffer, nBufferSize, blockDim.x, nKernelHalf };
        buffer.initializeCache(); /* loads first set of data */

        /* The for loop break condition is the same for all threads in a block,
         * so it is safe to use __syncthreads() inside */
        for ( T_PREC * curDataRow = data; curDataRow < data + rnDataX; curDataRow += nThreads )
        {
            buffer.loadCacheLine( curDataRow );

            /* calculated weighted sum on inner points in buffer, but only if
             * the value we are at is actually needed: */
            const unsigned iBuf = nKernelHalf + threadIdx.x;
            if ( &curDataRow[iBuf-nKernelHalf] < &data[rnDataX] )
            {
                T_PREC sum = T_PREC(0);
                for ( T_PREC * w = rWeights, * x = &buffer[iBuf-nKernelHalf];
                      w < rWeights + rnWeights; ++w, ++x )
                {
                    sum += (*w) * (*x);
                }
                /* write result back into memory (in-place). No need to wait for
                 * all threads to finish, because we write into global memory,
                 * to values we already buffered into shared memory! */
                curDataRow[iBuf-nKernelHalf] = sum;
            }
        }
    }



    template<class T_PREC>
    void cudaGaussianBlurHorizontalConstantWeights
    (
        T_PREC * const rdpData,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        double const rSigma,
        cudaStream_t rStream,
        bool const rAsync
    )
    {
        static unsigned firstFree = 0;
        static float kernelSigmas[ nMaxKernels ];
        static float kernelSizes [ nMaxKernels ];

        /* look if we already have that kernel buffered */
        unsigned iKernel = 0;
        for ( ; iKernel < firstFree; ++iKernel )
            if ( kernelSigmas[ iKernel ] == rSigma )
                break;

        T_PREC * dpWeights = NULL;
        unsigned kernelSize = 0;
        /* if not found, then calculate and save it */
        if ( iKernel == firstFree )
        {
            //printf("sigma = %f not found, uploading to constant memory\n", rSigma );

            /* calc kernel */
            T_PREC pKernel[nMaxWeights];
            kernelSize = ::imresh::libs::calcGaussianKernel( rSigma, (T_PREC*) pKernel, nMaxWeights );
            assert( kernelSize <= nMaxWeights );

            /* if buffer full, then delete buffer */
            if ( firstFree == nMaxKernels )
            {
                #ifndef NDEBUG
                    std::cout << "Warning, couldn't find sigma in kernel buffer and no space to store it. Clearing buffer!\n";
                #endif
                firstFree = 0;
                iKernel = 0;
            }

            /* remember sigma */
            kernelSigmas[ iKernel ] = rSigma;
            kernelSizes [ iKernel ] = kernelSize;
            ++firstFree;

            /* upload to GPU */
            CUDA_ERROR( cudaGetSymbolAddress( (void**) &dpWeights, gdpGaussianWeights ) );
            dpWeights += iKernel * nMaxWeights;
            CUDA_ERROR( cudaMemcpyAsync( dpWeights, pKernel,
                kernelSize * sizeof( pKernel[0] ), cudaMemcpyHostToDevice, rStream ) );
            //CUDA_ERROR( cudaStreamSynchronize( rStream ) ); // not necessary, because kernel goes to same stream
        }
        else
        {
            CUDA_ERROR( cudaGetSymbolAddress( (void**) &dpWeights, gdpGaussianWeights ) );
            dpWeights += iKernel * nMaxWeights;
            kernelSize = kernelSizes[ iKernel ];
        }

        /* the image must be at least nThreads threads wide, else many threads
         * will only sleep. The number of blocks is equal to the image height.
         * Every block works on 1 image line. The number of Threads is limited
         * by the hardware to be e.g. 512 or 1024. The reason for this is the
         * limited shared memory size! */
        const unsigned nThreads = 256;
        const unsigned nBlocks  = rnDataY;
        const unsigned bufferSize = nThreads + kernelSize-1;

        cudaKernelApplyKernelConstantWeights<<<
            nBlocks,nThreads,
            sizeof(T_PREC) * bufferSize,
            rStream
        >>>( rdpData, rnDataX, rnDataY, dpWeights, kernelSize );

        if ( not rAsync )
            CUDA_ERROR( cudaStreamSynchronize( rStream ) );
    }




    /* explicit template instantiations */
    template
    void cudaGaussianBlurHorizontalConstantWeights<float>
    (
        float * rdpData,
        unsigned int rnDataX,
        unsigned int rnDataY,
        double rSigma,
        cudaStream_t rStream,
        bool rAsync
    );


} // namespace cuda
} // namespace algorithms
} // namespace imresh
} // namespace benchmark
