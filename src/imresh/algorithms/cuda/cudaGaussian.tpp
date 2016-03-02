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
#include <cuda_to_cupla.hpp>
#include "libs/calcGaussianKernel.hpp"
#include "libs/cudacommon.hpp"


namespace imresh
{
namespace algorithms
{
namespace cuda
{


    /* can't use cmath because it must work in CUDA and in OpenMP and
     * everything, because we use cupla
     * @see https://github.com/ComputationalRadiationPhysics/cupla/issues/28 */
    #ifndef max
    #   define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
    #endif
    #ifndef min
    #   define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
    #endif

    /** this flag adds some debug output to stdout if set to 1 **/
    #define DEBUG_CUDAGAUSSIAN_CPP 0


    /**
     * Can reuse gaussian kernels already sent to the GPU.
     * not threadsafe !
     *
     * Problematic to generalize for arbitrary kernels, because they don't
     * have a nice key value like sigma. We would have to compare every
     * element. That still would possibly be faster than initiating a
     * cudaMemcpyHostToDevice transfer.
     *
     * @todo don't delete whole buffer if full, but only the oldest element
     **/
    template<class T_PREC>
    class GaussianKernelGpuBuffer
    {
    private:
        /**
         * if N := nMaxKernels set to 0 then buffering is deactivated. 1 means
         * that buffering will only work if gaussian blur is called multiple
         * times with the same sigma. Alternating sigmas will will result in
         * no buffering. Higher N will buffer more kernels.
         * - Note that higher N only sets the maximum amount of kernels. If only
         *   M < N kernels are ever used, then the effective buffered kernels
         *   will be M. This is relevant, because a search O(N) needs to be
         *   performed on the buffered kernels.
         * - N itself is only bound by GPU global memory, meaning
         *     nMaxKernels * kernelSizes * sizeof(T_PREC)
         *   should be smaller than 100MB to be negligible. This means e.g.
         *   for double and kernelSizes ~ 50 -> nMaxKernels = 250
         * - The shrinkWrap algorithm will use a fixed number of sigmas to
         *   convolve. The parameters mentioned in
         *   @see dx.doi.org/10.1103/PhysRevB.68.140101
         *   are: "The width sigma is set to 3 pixels in the first iteration,
         *   and reduced by 1% every 20 iterations down to a minimum of 1.5"
         *   This makes 1.5 = 3.0*0.99^n => n = log(0.5)/log(0.99) = 69
         *   Buffering all these kernels will take not more than 50MB and
         *   thereby should be done.
         **/
        static constexpr unsigned mnMaxKernels   = 128;
        /**
         * if the kernel size is larger than this value, then it won't be
         * buffered.
         * - For the normal shrinkWrap the largest sigma will be 3. The kernel
         *   size for that is 19 elements. Meaning a value of 32 should be ok
         *   for most purposes and also would be aligned relatively well.
         **/
        static constexpr unsigned mMaxKernelSize = 32;

        struct DeviceGaussianKernels
        {
            /* array of length mnMaxKernels * mMaxKernelSize * sizeof(T_PREC) */
            T_PREC * dpKernelBuffer;
            /* first free kernel space */
            unsigned firstFree;
            /* use std::map for O(log N) search on the key == sigma and the value
             * being an index which can be used for mKernelSizes and to calculate
             * the offset from mdpKernelBuffer */
            std::map<T_PREC,int> kernelSigmas;
        };

        /* list of ( iDevice, bufferStruct ) tuples */
        std::list< std::pair< int, DeviceGaussianKernels > > mDeviceBuffers;
        /* it would be bad, if we already had added some sigma value, for the
         * other thread to read, but the data wasn't uploaded successfully yet.
         * Also we don't want other threads to think that sigma isn't available
         * while another thread is adding it, meaning two threads will add
         * that kernel */
        std::mutex mBuffersMutex;

        /* not thread-safe! call mBuffersMutex.lock(); prior to calling this */
        DeviceGaussianKernels & getDeviceBuffer ( void )
        {
            int curDevice;
            CUDA_ERROR( cudaGetDevice( &curDevice ) );

            for ( auto & deviceTuple : mDeviceBuffers )
            {
                if ( deviceTuple.first == curDevice )
                    return deviceTuple.second;
            }

            /* if no match was found and returned, then create new buffer
             * on new GPU */

            DeviceGaussianKernels buffer;
            buffer.dpKernelBuffer = NULL;
            CUDA_ERROR( cudaMalloc( (void**) &buffer.dpKernelBuffer, mnMaxKernels *
                mMaxKernelSize * sizeof( buffer.dpKernelBuffer[0] ) ) );
            buffer.firstFree = 0;
            assert( mnMaxKernels <= buffer.kernelSigmas.max_size() );

            mDeviceBuffers.push_back( { curDevice, buffer } ); // !copy into list
            auto & added = mDeviceBuffers.back().second;

            #if DEBUG_CUDAGAUSSIAN_CPP == 1
                std::cout << "[Note] GaussianKernelGpuBuffer no buffer for device found, created new one at " << buffer.dpKernelBuffer << " in global memory on device" << curDevice << std::endl;
            #endif
            return added;
        }

        GaussianKernelGpuBuffer() {}; /* forbid construction except from itself */
        GaussianKernelGpuBuffer( const GaussianKernelGpuBuffer & ); /* forbid copy */
        GaussianKernelGpuBuffer & operator=( const GaussianKernelGpuBuffer & ); /* ibid */

    public:

        static GaussianKernelGpuBuffer & getInstance()
        {
            /* This will be thread safe since C++11, where static variables
             * is always thread-safe (according to section 6.7 of The Standard) */
            static GaussianKernelGpuBuffer mInstance;
            return mInstance;
        }

        /* Destructor */
        ~GaussianKernelGpuBuffer()
        {
            #if DEBUG_CUDAGAUSSIAN_CPP == 1
                std::cout << "[Note] GaussianKernelGpuBuffer Destructor called.\n" << std::endl;
            #endif
            int oldDevice;
            CUDA_ERROR( cudaGetDevice( &oldDevice ) );
            for ( auto & deviceTuple : mDeviceBuffers )
            {
                CUDA_ERROR( cudaSetDevice( deviceTuple.first ) );
                if ( deviceTuple.second.dpKernelBuffer != NULL )
                {
                    CUDA_ERROR( cudaFree( deviceTuple.second.dpKernelBuffer ) );
                    deviceTuple.second.dpKernelBuffer = NULL;
                }
            }
            CUDA_ERROR( cudaSetDevice( oldDevice ) );
        }

        /**
         * @param[in]  rSigma
         * @param[out] rdppKernel will contain the pointer to the pointer to
         *             device memory containing the kernel
         * @param[out] rpKernelSize will contain the number of elements of the
         *             kernel
         * @param[in]  rStream CUDA stream to use
         * @param[in]  rAsync if true, then don't wait for the CUDA kernel to
         *             finish, else call cudaStreamSynchronize on rStream.
         **/
        void getGpuPointer
        (
            T_PREC const & rSigma,
            T_PREC* * rdppKernel,
            unsigned int * rpKernelSize,
            cudaStream_t rStream = 0,
            bool const rAsync = true
        )
        {
            #if DEBUG_CUDAGAUSSIAN_CPP == 1
                std::cout << "[Note] GaussianKernelGpuBuffer::getGpuPointer called.\n" << std::endl;
            #endif
            mBuffersMutex.lock();

            auto & buffer = getDeviceBuffer();

            /* look if we already have that kernel buffered */
            /* The single element versions (1) return a pair, with its member pair::first set to an iterator pointing to either the newly inserted element or to the element with an equivalent key in the map. The pair::second element in the pair is set to true if a new element was inserted or false if an equivalent key already existed. */
            /* In this case the return value is:
             * std::pair<std::map<T_PREC,int>::iterator,bool> */
            auto inserted = buffer.kernelSigmas.insert( std::pair<T_PREC,int>( rSigma, buffer.firstFree ) );

            assert( inserted.first->first  == rSigma );
            if ( inserted.second == true )
                assert( inserted.first->second == (int) buffer.firstFree );

            auto & iKernel = inserted.first->second;
            /* calculate return values. If there are problems they will be
             * changed to more correct values, e.g. if the kernel doesn't fit
             * into the standard buffer */
            *rdppKernel = buffer.dpKernelBuffer + iKernel * mMaxKernelSize;
            unsigned kernelSize = libs::calcGaussianKernel( rSigma, (T_PREC*) NULL, 0 );
            *rpKernelSize = kernelSize;

            /* if kernel not found in buffer, then calculate and upload it */
            if ( inserted.second == true )
            {
                //printf("sigma = %f not found, uploading to global memory\n", rSigma );

                /* calc kernel to pKernel */
                T_PREC pKernel[mMaxKernelSize];
                kernelSize = libs::calcGaussianKernel( rSigma, (T_PREC*) pKernel, mMaxKernelSize );
                assert( kernelSize > 0 );

                /* if kernel fits into buffer */
                if ( kernelSize <= mMaxKernelSize )
                {
                    /* if buffer full, then clear buffer */
                    if ( iKernel == (int) mnMaxKernels )
                    {
                        #ifndef NDEBUG
                            std::cout << "Warning, couldn't find sigma in kernel buffer and no space to store it. Clearing buffer completely! In order to avoid this increase mnMaxKernels in " << __FILE__ << "\n";
                        #endif
                        buffer.firstFree = 0;
                        iKernel = 0;
                        buffer.kernelSigmas.clear();
                        buffer.kernelSigmas.insert( std::pair<T_PREC,int>( rSigma, buffer.firstFree ) );
                    }

                    /* remember kernel size */
                    ++buffer.firstFree;

                    /* upload to GPU */
                    *rdppKernel = buffer.dpKernelBuffer + iKernel * mMaxKernelSize;
                    CUDA_ERROR( cudaMemcpyAsync( *rdppKernel, pKernel,
                        kernelSize * sizeof( pKernel[0] ), cudaMemcpyHostToDevice, rStream ) );
                    if ( not rAsync )
                        CUDA_ERROR( cudaStreamSynchronize( rStream ) );
                }
                /* if the kernel size doesn't fit into the buffer, we need to
                 * upload it unbuffered */
                else
                {
                    std::cout << "[ERROR] The kernel size " << kernelSize << " is larger than the default Gaussian kernel buffer kernel size of " << mMaxKernelSize << " (rSigma=" << rSigma << "). Because of thread-safety this currently is a fatal error and program limtation! " << __FILE__ << ":" << __LINE__ << std::endl;
                    mBuffersMutex.unlock(); /* not sure if other threads just continue :S and would hang */
                    abort();
                }
            }

            mBuffersMutex.unlock();
        }
    };
    template<class T_PREC> constexpr unsigned GaussianKernelGpuBuffer<T_PREC>::mnMaxKernels;
    template<class T_PREC> constexpr unsigned GaussianKernelGpuBuffer<T_PREC>::mMaxKernelSize;


    template<class T>
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY inline T * ptrMin ( T * const a, T * const b )
    {
        return a < b ? a : b;
    }
    /**
     * Provides a class for a moving window type 2d cache
     **/
    template< class T_ACC, class T_PREC >
    struct Cache1d
    {
        T_ACC const & acc;
        T_PREC const * const & data;
        unsigned int const & nData;

        T_PREC * const & smBuffer; /**< pointer to allocated buffer, will not be allocated on constructor because this class needs to be trivial to work on GPU */
        unsigned int const & nBuffer;

        unsigned int const & nThreads;
        unsigned int const & nKernelHalf;

        ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY inline T_PREC & operator[]( unsigned i ) const
        {
            return smBuffer[i];
        }

        #ifndef NDEBUG
        #if DEBUG_CUDAGAUSSIAN_CPP == 1
            ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY void printCache( void ) const
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

        ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY inline void initializeCache( void ) const
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

        ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY inline void loadCacheLine( T_PREC const * const curDataRow ) const
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

    /**
     * Choose the buffer size, so that in every step rnThreads data values
     * can be saved back and newly loaded. As we need N neighbors left and
     * right for the calculation of one value, especially at the borders,
     * this means, the buffer size needs to be rnThreads + 2*N elements long:
     * @verbatim
     *                                                   kernel
     * +--+--+--+--+--+--+--+--+--+--+--+--+        +--+--+--+--+--+
     * |xx|xx|  |  |  |  |  |  |  |  |yy|yy|        |  |  |  |  |  |
     * +--+--+--+--+--+--+--+--+--+--+--+--+        +--+--+--+--+--+
     * <-----><---------------------><----->        <-------------->
     *   N=2       rnThreads = 8      N=2             rnWeights = 5
     *                                              <----->  <----->
     *                                                N=2      N=2
     * @endverbatim
     * Elements marked with xx and yy can't be calculated, the other elements
     * can be calculated in parallel.
     *
     * In the first step the elements marked with xx are copie filled with
     * the value in the element right beside it, i.e. extended borders.
     *
     * In the step thereafter especially the elements marked yy need to be
     * calculated (if the are not already on the border). To calculate those
     * we need to move yy and N=2 elements to the left to the beginning of
     * the buffer and fill the rest with new data from rData:
     * @verbatim
     *               ((bufferSize-1)-(2*N-1)
     *                           |
     * <------------ bufferSize -v--------->
     * +--+--+--+--+--+--+--+--+--+--+--+--+
     * |xx|xx|  |  |  |  |  |  |  |  |yy|yy|
     * +--+--+--+--+--+--+--+--+--+--+--+--+
     *                         <----------->
     * <----------->                2*N=4
     *       ^                        |
     *       |________________________|
     *
     * +--+--+--+--+--+--+--+--+--+--+--+--+
     * |vv|vv|yy|yy|  |  |  |  |  |  |ww|ww|
     * +--+--+--+--+--+--+--+--+--+--+--+--+
     * <-----><---------------------><----->
     *   N=2       rnThreads = 8      N=2
     * @endverbatim
     * All elements except those marked vv and ww can now be calculated
     * in parallel. The elements marked yy are the old elements from the right
     * border, which were only used readingly up till now. The move of the
     * 2*N elements may be preventable by using a modulo address access, but
     * a move in shared memory / cache is much faster than waiting for the
     * rest of the array to be filled with new data from global i.e. uncached
     * memory.
     *
     * param[in] blockDim.x number of threads will be interpreted as how many
     *           values are to be calculated in parallel. The internal buffer
     *           stores then blockDim.x + 2*N values per step
     * param[in] blockDim.y number of rows to blur. threadIdx.y == 0 will blur
     *           rdpData[ 0...rImageWidth-1 ], threadIdx.y == 1 the next
     *           rImageWidth elements. Beware that you must not start more
     *           threads in y direction than the image has rows, else a
     *           segfault will occur!
     * param[in] nKernelHalf kernel half size, meaning the kernel is supposed
     *           to be 2*N+1 elements long. N can also be interpreted as the
     *           number of neighbors in each direction needed to calculate one
     *           value.
     **/
    template<class T_PREC>
    struct cudaKernelApplyKernelSharedWeights
    {
    template< class T_ACC >
    ALPAKA_FN_NO_INLINE_ACC
    void operator()
    (
        T_ACC const & acc,
        /* You can't pass by reference to a kernel !!! compiles, but gives weird errors ... */
        T_PREC * const rdpData,
        unsigned int const rImageWidth,
        T_PREC const * const rdpWeights,
        unsigned int const nKernelHalf
    ) const
    {
        assert( blockDim.y == 1 and blockDim.z == 1 );
        assert(  gridDim.y == 1 and  gridDim.z == 1 );

        /* If more than 1 block, then each block works on a separate line.
         * Each line borders will be extended. So mutliple blocks can't be used
         * to blur one very very long line even faster! */
        int const & nThreads = blockDim.x;
        T_PREC * const data = &rdpData[ blockIdx.x * rImageWidth ];

        /* manage dynamically allocated shared memory */
        /* @see http://stackoverflow.com/questions/27570552/ */
        sharedMemExtern( dynamicSharedMemory, unsigned char );
        T_PREC * const smBlock = reinterpret_cast<T_PREC*>( dynamicSharedMemory );

        unsigned int const nWeights = 2*nKernelHalf+1;
        unsigned int const nBufferSize = nThreads + 2*nKernelHalf;
        T_PREC * const smWeights = smBlock;
        T_PREC * const smBuffer  = &smBlock[ nWeights ];
        __syncthreads();

        Cache1d<T_ACC, T_PREC> buffer{ acc, data, rImageWidth, smBuffer, nBufferSize, blockDim.x, nKernelHalf };

        /* cache the weights to shared memory. Benchmarks imageSize 1024x1024
         * parallel (pointer arithmetic) : 0.95ms
         * parallel                      : 1.1ms
         * memcpy                        : 1.57ms
        */
        #if true
            {
                T_PREC * target    = smWeights  + threadIdx.x;
                T_PREC const * src = rdpWeights + threadIdx.x;
                for ( ; target < smWeights + nWeights;
                     target += blockDim.x, src += blockDim.x )
                {
                    *target = *src;
                }
            }
        #elif false
            for ( unsigned iWeight = threadIdx.x; iWeight < nWeights; ++iWeight )
                smWeights[iWeight] = rdpWeights[iWeight];
        #else
            if ( threadIdx.x == 0 )
                memcpy( smWeights, rdpWeights, sizeof(T_PREC)*nWeights );
        #endif


        #ifndef NDEBUG
        #if DEBUG_CUDAGAUSSIAN_CPP == 1
            if ( blockIdx.x == 0 and threadIdx.x == 0 )
            {
                printf( "================ cudaGaussianApplyKernel ================\n" );
                printf( "\gridDim = (%i,%i,%i), blockDim = (%i,%i,%i)\n",
                        gridDim.x, gridDim.y, gridDim.z,
                        blockDim.x, blockDim.y, blockDim.z );
                printf( "rImageWidth = %u\n", rImageWidth );
                printf( "\nConvolve Kernel : \n");
                for ( unsigned iW = 0; iW < nWeights; ++iW )
                    printf( "%10.6f ", smWeights[iW] );
                printf( "\n" );

                printf( "\nInput Vector to convolve horizontally : \n" );
                for ( unsigned i = 0; i < rImageWidth; ++i )
                    printf( "%10.6f ", data[ i ] );
                printf( "\n" );
            }
        #endif
        #endif

        buffer.initializeCache();

        /* Loop over buffers. If rnData == rnThreads then the buffer will
         * exactly suffice, meaning the loop will only be run 1 time.
         * The for loop break condition is the same for all threads, so it is
         * safe to use __syncthreads() inside */
        for ( T_PREC * curDataRow = data; curDataRow < data + rImageWidth; curDataRow += nThreads )
        {
            buffer.loadCacheLine( curDataRow );
            #ifndef NDEBUG
            #if DEBUG_CUDAGAUSSIAN_CPP == 1
                if ( rImageWidth == 2 )
                    return; //assert(false);
            #endif
            #endif

            /* calculated weighted sum on inner points in buffer, but only if
             * the value we are at is actually needed: */
            const unsigned iBuf = nKernelHalf + threadIdx.x;
            if ( &curDataRow[iBuf-nKernelHalf] < &data[rImageWidth] )
            {
                T_PREC sum = T_PREC(0);
                /* this for loop is done by each thread and should for large
                 * enough kernel sizes sufficiently utilize raw computing power */
                for ( T_PREC * w = smWeights, * x = &buffer[iBuf-nKernelHalf];
                      w < &smWeights[nWeights]; ++w, ++x )
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
    };

    template<class T_PREC>
    void cudaGaussianBlurHorizontalSharedWeights
    (
        T_PREC * const rdpData,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        double const rSigma,
        cudaStream_t rStream,
        bool rAsync
    )
    {
        /**
         * Object which manages kernels sent to the GPU to possibly reuse them
         *
         * This is especially useful for shrinkWrap, because every shrinkWrap
         * call uses identical kernels! Meaning after the first shrinkWrap call
         * all other won't need to send kernels to the GPU anymore, thereby
         * reducing latency quite a bit!
         *
         * @verbatim
         *   Image Size  : non buffered kernel | buffered kernel
         *   (1,1)       :  0.200704           |  0.021888
         *   (2,2)       :  0.200128           |  0.026464
         *   (2,2)       :  0.200384           |  0.021952
         *   (4,4)       :  0.201024           |  0.021984
         *   (6,6)       :   0.20064           |  0.021024
         *   (8,8)       :   0.20368           |  0.022048
         *   (12,12)     :  0.201472           |  0.026336
         *   (18,18)     :  0.200512           |  0.026432
         *   (25,25)     :   0.20672           |  0.022432
         *   (37,37)     :   0.20304           |  0.034592
         *   (53,53)     :  0.211648           |  0.025184
         *   (77,77)     :  0.212096           |  0.033664
         *   (111,111)   :  0.193696           |  0.033152
         *   (160,160)   :  0.202464           |  0.040992
         *   (230,230)   :    0.2448           |   0.05552
         *   (331,331)   :  0.240096           |   0.10464
         *   (477,477)   :  0.300096           |  0.149664
         *   (687,687)   :   0.45472           |  0.271904
         *   (989,989)   :   0.67296           |  0.477888
         *   (2048,2048) :   1.84432           |   1.72842
         * @endverbatim
         *
         * The effect mainly is visible for sizes smaller than 1000x1000,
         * because it is only an offset (It doesn't scale with the problem
         * size)
         **/

        unsigned kernelSize;
        T_PREC * dpKernel;
        GaussianKernelGpuBuffer<T_PREC>::getInstance().getGpuPointer( rSigma, &dpKernel, &kernelSize, rStream, true /* async, goes to same stream */ );

        /* the image must be at least nThreads threads wide, else many threads
         * will only sleep. The number of blocks is equal to the image height.
         * Every block works on 1 image line. The number of Threads is limited
         * by the hardware to be e.g. 512 or 1024. The reason for this is the
         * limited shared memory size! */
        const unsigned nThreads = 256;
        const unsigned nBlocks  = rnDataY;
        const unsigned N = (kernelSize-1)/2;
        const unsigned bufferSize = nThreads + 2*N;

        CUPLA_KERNEL( cudaKernelApplyKernelSharedWeights<T_PREC> )(
            nBlocks,nThreads,
            sizeof(T_PREC)*( kernelSize + bufferSize ),
            rStream
        )( rdpData, rnDataX, dpKernel, N );

        if ( not rAsync )
            CUDA_ERROR( cudaStreamSynchronize( rStream ) );
    }



    /**
     * Calculates the weighted sum vertically i.e. over the rows.
     *
     * In order to make use of Cache Lines blockDim.x columns are always
     * calculated in parallel. Furthermore to increase parallelism blockIdx.y
     * threads can calculate the values for 1 column in parallel:
     * @verbatim
     *                gridDim.x=3
     *               <---------->
     *               blockDim.x=4
     *                    <-->
     *            I  #### #### ## ^
     *            m  #### #### ## | blockDim.y
     *            a  #### #### ## v    = 3
     *            g  #### #### ## ^
     *            e  #### #### ## | blockDim.y
     *                            v    = 3
     *               <---------->
     *               imageWidth=10
     * @endverbatim
     * The blockIdx.y threads act as a sliding window. Meaning in the above
     * example y-thread 0 and 1 need to calculate 2 values per kernel run,
     * y-thread 2 only needs to calculate 1 calue, because the image height
     * is not a multiplie of blockIdx.y
     *
     * Every block uses a shared memory buffer which holds roughly
     * blockDim.x*blockDim.y elements. In order to work on wider images the
     * kernel can be called with blockDim.x != 0
     *
     * @see cudaKernelApplyKernel @see gaussianBlurVertical
     *
     * param[in] N kernel half size. rdpWeights is assumed to be 2*N+1
     *           elements long.
     * param[in] blockDim.x number of columns to calculate in parallel.
     *           this should be a value which makes full use of a cache line,
     *           i.e. 32 Warps * 4 Byte = 128 Byte for a NVidia GPU (2016)
     * param[in] blockDim.y number of rows to calculate in parallel.
     *           This value shouldn't be too small, because else we are only
     *           moving the buffer date to and fro without doing much
     *           calculation. That happens because of the number of neighbors
     *           N needed to calculate 1 value. If the buffer is 2*N+1
     *           elements large ( blockDim.y == 1 ), then we can only
     *           calculate 1 value with that buffer data.
     *           @todo implement buffer index modulo instead of shifting the
     *                 values in memory
     **/
    template<class T_PREC>
    struct cudaKernelApplyKernelVertically
    {
    template< class T_ACC >
    ALPAKA_FN_NO_INLINE_ACC
    void operator()
    (
        T_ACC const & acc,
        T_PREC * const rdpData,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        T_PREC const * const rdpWeights,
        unsigned int const N
    ) const
    {
        assert( blockDim.z == 1 );
        assert( gridDim.y == 1 and  gridDim.z == 1 );

        const unsigned linThreadId = threadIdx.y * blockDim.x + threadIdx.x;
        const unsigned nThreads = blockDim.x * blockDim.y;

        /* the shared memory buffer dimensions */
        const unsigned nColsCacheLine = blockDim.x;
        const unsigned nRowsCacheLine = blockDim.y + 2*N;

        /* Each block works on a separate group of columns */
        const unsigned iCol = blockIdx.x * blockDim.x + threadIdx.x;
        T_PREC * const data = rdpData + iCol;
        /* the rightmost block might not be full. In that case we need to mask
         * those threads working on the columns right of the image border */
        const bool iAmSleeping = iCol >= rnDataX;

        /* The dynamically allocated shared memory buffer will fit the weights and
         * the values to calculate + the 2*N neighbors needed to calculate them */
        sharedMemExtern( dynamicSharedMemory, unsigned char );
        T_PREC * const smBlock = reinterpret_cast<T_PREC*>( dynamicSharedMemory );

        const unsigned nWeights    = 2*N+1;
        const unsigned nBufferSize = nColsCacheLine * nRowsCacheLine;
        T_PREC * const smWeights   = smBlock;
        T_PREC * const smBuffer    = smBlock + nWeights;

        /* cache the weights to shared memory */
        __syncthreads();
        for ( int i = linThreadId; i < nWeights; i += nThreads )
            smWeights[i] = rdpWeights[i];

        /**
         * @verbatim
         *                        (shared memory)
         *                         kernel (size 3)
         *                      +------+-----+-----+
         *                      | w_-1 | w_0 | w_1 |
         *                      +------+-----+-----+
         *       (global memory)
         *       data to convolve
         *    +------+------+------+------+    (should be a multiple of
         *    | a_00 | a_01 | a_02 | a_02 |   cache line wide i.e. nRows)
         *    +------+------+------+------+        (shared memory)
         *    | a_10 | a_11 | a_12 | a_12 |         result buffer
         *    +------+------+------+------+        +------+------+  |
         *    | a_20 | a_21 | a_22 | a_22 |        | b_00 | b_01 |  |
         *    +------+------+------+------+        +------+------+  | threadIdx
         *    | a_30 | a_31 | a_32 | a_32 |        | b_10 | b_11 |  |    .y
         *    +------+------+------+------+        +------+------+  |
         *    | a_40 | a_41 | a_42 | a_42 |        | b_20 | b_21 |  |
         *    +------+------+------+------+        +------+------+  v
         *    | a_50 | a_51 | a_52 | a_52 |        -------------->
         *    +------+------+------+------+          threadIdx.x
         *    <-------------><------------>
         *      threadIdx.x    threadIdx.x
         *    <--------------------------->
         *             blockIdx.x
         *
         *        b_0x = w_-1*a_1x + w_0*a_2x + w_1*a_3x
         *        b_1x = w_-1*a_2x + w_0*a_3x + w_1*a_4x
         *        b_1x = w_-1*a_3x + w_0*a_4x + w_1*a_5x
         *        b_1x = w_-1*a_3x + w_0*a_4x + w_1*a_5x
         * @endverbatim
         * In order to reduce global memory accesses, we can reorder the
         * calculation of b_ij so that we can cache one row of a_ij and basically
         * broadcast ist to b_ij:
         *
         *  a) cache a_1x  ->  b_0x += w_-1*a_1x
         *  b) cache a_2x  ->  b_0x += w_0*a_2x, b_1x += w_-1*a_2x
         *  c) cache a_3x  ->  b_0x += w_1*a_3x, b_1x += w_0*a_3x, b_2x += w_-1*a_3x
         *  d) cache a_4x  ->                    b_1x += w_1*a_1x, b_2x += w_0*a_4x
         *  e) cache a_5x  ->                                      b_2x += w_1*a_5x
         *
         * The buffer size needs at least kernelSize rows. If it's equal to kernel
         * size rows, then in every step one row will be completed calculating,
         * meaning it can be written back.
         * This enables us to use a round-robin like calculation:
         *   - after step c we can write-back b_0x to a_3x, we don't need a_3x
         *     anymore after this step.
         *   - because the write-back needs time the next calculation in d should
         *     write to b_1x. This is the case
         *   - the last operation in d would then be an addition to b_3x == b_0x
         *     (use i % 3)
         **/

        /* In the first step extend upper border. Write them into the N elements
         * before the lower border-N, beacause in the first step in the loop
         * these elements will be moved to the upper border, see below. */
        T_PREC * const smTargetRow = &smBuffer[ nBufferSize - 2*N*nColsCacheLine ];
        #ifdef GAUSSIAN_PERIODIC
            if ( not iAmSleeping )
            {
                for ( unsigned iRow = threadIdx.y; iRow < N; iRow += blockDim.y )
                {
                    smTargetRow[ iRow * nColsCacheLine ] =
                    data[ ( iRow % rnDataY ) * nColsCacheLine ];
                }
            }
        #else
        {
            const T_PREC upperBorderValue = *data;
            for ( auto pTarget = smTargetRow + linThreadId;
                  pTarget < smTargetRow + N * nColsCacheLine;
                  pTarget += nThreads )
            {
                *pTarget = upperBorderValue;
            }
        }
        #endif


        /* Loop over and calculate the rows. If rnDataY == blockDim.y, then the
         * buffer will exactly suffice, meaning the loop will only be run 1 time */
        for ( unsigned iRow = 0; iRow < rnDataY; iRow += blockDim.y )
        {
            T_PREC * const curDataRow = data + ( iRow + threadIdx.y ) * rnDataX;
            /* move last N rows to the front of the buffer */
            __syncthreads();
            /* memcpy( smBuffer, smTargetRow, N*nColsCacheLine*sizeof(smBuffer[0]) ); */
            for ( T_PREC * pTarget = smBuffer    + linThreadId,
                         * pSource = smTargetRow + linThreadId;
                  pTarget < smBuffer + N * nColsCacheLine;
                  pTarget += nThreads, pSource += nThreads )
            {
                *pTarget = *pSource;
            }

            /* Load blockDim.y + N rows into buffer.
             * If data end reached, fill buffer rows with last row */
            #if false
            if ( not iAmSleeping )
            {
                T_PREC * pTarget = smBuffer +
                    N * nColsCacheLine /*skip first N rows*/ + linThreadId;
                for ( unsigned curRow = threadIdx.y;
                      pTarget < smBuffer + nBufferSize;
                      pTarget += blockDim.y * nColsCacheLine, curRow += blockDim.y )
                {
                    #ifdef GAUSSIAN_PERIODIC
                        const int iRowMod = curRow % rnDataY;
                    #else
                        const int iRowMod = min( curRow, rnDataY-1 );
                    #endif
                    *pTarget = data[ iRowMod*rnDataX ];
                }
            }
            #else
                /*   a) Load blockDim.y rows in parallel */
                T_PREC * const pLastData = &data[ (rnDataY-1)*rnDataX ];
                const unsigned iBuf = /*skip first N rows*/ N * nColsCacheLine
                                    + threadIdx.y * nColsCacheLine + threadIdx.x;
                __syncthreads();
                if ( not iAmSleeping )
                {
                    T_PREC * const datum = ptrMin(
                        curDataRow,
                        pLastData
                    );
                    assert( iBuf < nBufferSize );
                    smBuffer[iBuf] = *datum;
                }
                /*   b) Load N rows by master threads, because nThreads >= N is not
                 *      guaranteed. */
                    if ( not iAmSleeping and threadIdx.y == 0 )
                    {
                        for ( unsigned iBufRow = N+blockDim.y; iBufRow < nRowsCacheLine; ++iBufRow )
                        {
                            T_PREC * const datum = ptrMin(
                                &data[ (iRow+iBufRow-N) * rnDataX ],
                                pLastData
                            );
                            const unsigned iBuffer = iBufRow*nColsCacheLine + threadIdx.x;
                                assert( iBuffer < nBufferSize );
                            smBuffer[ iBuffer ] = *datum;
                        }
                    }
            #endif
            __syncthreads();

            /* calculated weighted sum on inner rows in buffer, but only if
             * the value we are at is inside the image */
            if ( ( not iAmSleeping ) and iRow+threadIdx.y < rnDataY )
            {
                T_PREC sum = T_PREC(0);
                /* this for loop is done by each thread and should for large
                 * enough kernel sizes sufficiently utilize raw computing power */
                T_PREC * w = smWeights;
                T_PREC * x = &smBuffer[ threadIdx.y * nColsCacheLine + threadIdx.x ];
                #pragma unroll
                for ( ; w < &smWeights[nWeights]; ++w, x += nColsCacheLine )
                {
                    assert( w < smWeights + nWeights );
                    assert( x < smBuffer + nBufferSize );
                    sum += (*w) * (*x);
                }
                /* write result back into memory (in-place). No need to wait for
                 * all threads to finish, because we write into global memory, to
                 * values we already buffered into shared memory! */
                assert( curDataRow < &rdpData[ rnDataX * rnDataY ] );
                *curDataRow = sum;
            }
        }

    }
    };


    template<class T_PREC>
    void cudaGaussianBlurVertical
    (
        T_PREC * const rdpData,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        double const rSigma,
        cudaStream_t rStream,
        bool rAsync
    )
    {
        unsigned kernelSize;
        T_PREC * dpKernel;
        GaussianKernelGpuBuffer<T_PREC>::getInstance().getGpuPointer( rSigma, &dpKernel, &kernelSize, rStream, true /* async, goes to same stream */ );

        /**
         *  - The image should be at least nThreadsX threads wide, else many
         *    threads will only sleep. The number of blocks is
         *      ceil( image height / nThreadsX )
         *    Every block works on nThreadsX image columns.
         *  - blockDim.x * gridDim.x must be larger or equal to the image size
         *    or else only that amount of image columns will be blurred
         *  - Those columns use nThreadsY threads to parallelize the
         *    calculation per column. nThreadsY should be large compared to
         *    kernelSize, or else the kernel will spent most of its time
         *    shifting data inside the shared memory
         *    @todo implement index cycling instead of shifting
         *  - The number of Threads is limited by the hardware, but more
         *    imminently by the shared memory being used, @see bufferSize
         *  - nThreadsX should be a multiple of a cache line / superword =
         *    32 warps * 1 float per warp = 128 Byte => nThreadsX = 32.
         *    For double 16 would also suffice.
         **/
        dim3 nThreads( 32, 1024/32, 1 );
        dim3 nBlocks ( 1, 1, 1 );
        nBlocks.x = (unsigned) ceilf( (float) rnDataX / nThreads.x );
        const unsigned kernelHalfSize = (kernelSize-1)/2;
        const unsigned bufferSize     = nThreads.x*( nThreads.y + 2*kernelHalfSize );

        CUPLA_KERNEL( cudaKernelApplyKernelVertically<T_PREC> )(
            nBlocks,nThreads,
            sizeof( dpKernel[0] ) * ( kernelSize + bufferSize ),
            rStream
        )( rdpData, rnDataX, rnDataY, dpKernel, kernelHalfSize );

        if ( not rAsync )
            CUDA_ERROR( cudaStreamSynchronize( rStream ) );
    }


    template<class T_PREC>
    void cudaGaussianBlur
    (
        T_PREC * const rdpData,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        double const rSigma,
        cudaStream_t rStream,
        bool rAsync
    )
    {
        cudaGaussianBlurHorizontal<T_PREC>( rdpData,rnDataX,rnDataY,rSigma, rStream,rAsync );
        cudaGaussianBlurVertical  <T_PREC>( rdpData,rnDataX,rnDataY,rSigma, rStream,rAsync );
    }


    template<class T_PREC>
    void cudaGaussianBlurSharedWeights
    (
        T_PREC * const rdpData,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        double const rSigma,
        cudaStream_t rStream,
        bool rAsync
    )
    {
        cudaGaussianBlurHorizontalSharedWeights( rdpData,rnDataX,rnDataY,rSigma, rStream,rAsync );
        cudaGaussianBlurVertical               ( rdpData,rnDataX,rnDataY,rSigma, rStream,rAsync );
    }


} // namespace cuda
} // namespace algorithms
} // namespace imresh
