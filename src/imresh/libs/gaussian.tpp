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


#include "gaussian.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <type_traits>
#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif
#include <cassert>
#include <cstring>  // memcpy, memset
#include <cstddef>  // NULL
#include <cstdlib>  // malloc, free
#ifdef USE_FFTW
#   include <fftw3.h>
#endif

#include "calcGaussianKernel.hpp"


namespace imresh
{
namespace libs
{


    #define DEBUG_GAUSSIAN_CPP 0


    template<class T_Prec>
    void applyKernel
    (
        T_Prec *       const rData    ,
        unsigned int   const rnData   ,
        const T_Prec * const rWeights ,
        unsigned int   const rnWeights,
        unsigned int   const rnThreads
    )
    {
        assert( rnWeights > 0 );
        assert( rnWeights % 2 == 1 );
        assert( rnThreads > 0 );
        /**
         *      kernel
         * +--+--+--+--+--+
         * |  |  |  |  |  |
         * +--+--+--+--+--+
         * <-------------->
         *   rnWeights = 5
         * <----->  <----->
         *   N=2      N=2
         **/
        const unsigned N = (rnWeights-1)/2;

        /**
         * Choose the buffer size, so that in every step rnThreads data values
         * can be saved back and newly loaded. As we need N neighbors left and
         * right for the calculation of one value, especially at the borders,
         * this means, the buffer size needs to be rnThreads + 2*N elements
         * long:
         *
         * +--+--+--+--+--+--+--+--+--+--+--+--+
         * |xx|xx|  |  |  |  |  |  |  |  |yy|yy|
         * +--+--+--+--+--+--+--+--+--+--+--+--+
         * <-----><---------------------><----->
         *   N=2       rnThreads = 8      N=2
         *
         * Elements marked with xx and yy can't be calculated, the other
         * elements can be calculated in parallel.
         *
         * In the first step the elements marked with xx are copie filled with
         * the value in the element right beside it, i.e. extended borders.
         *
         * In the step thereafter especially the elements marked yy need to be
         * calculated (if the are not already on the border). To calculate
         * those we need to move yy and N=2 elements to the left to the
         * beginning of the buffer and fill the rest with new data from rData:
         *
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
         *
         * All elements except those marked vv and ww can now be calculated
         * in parallel. The elements marked yy are the old elements from the
         * right border, which were only used readingly up till now. The move
         * of the 2*N elements may be preventable by using a modulo address
         * access, but a move in shared memory / cache is much faster than
         * waiting for the rest of the array to be filled with new data from
         * global i.e. uncached memory.
         **/
        const int bufferSize = rnThreads + 2*N;
        T_Prec * buffer = (T_Prec*) malloc( sizeof(T_Prec)*bufferSize );

        /* In the first step initialize the left border to the same values (extend) */
        const T_Prec leftBorderValue = rData[0];
        for ( unsigned iB = 0; iB < N; ++iB )
            buffer[ bufferSize-2*N+iB ] = leftBorderValue;

        /* Loop over buffers. If rnData == rnThreads then the buffer will
         * exactly suffice, meaning the loop will only be run 1 time */
        for ( T_Prec * dataPos = rData; dataPos < &rData[rnData]; dataPos += rnThreads )
        {
            /* move last N elements to the front of the buffer */
            /* __syncthreads(); */
            for ( unsigned iB = 0; iB < N; ++iB )
                buffer[iB] = buffer[ bufferSize-2*N+iB ];

            /* Load rnThreads+N data elements into buffer. If data end reached,
             * fill buffer with last data element */
            /* __syncthreads(); */
            #pragma omp parallel for
            for ( unsigned iB = N; iB < rnThreads+2*N; ++iB )
                if ( &dataPos[ iB-N ] < &rData[ rnData ] )
                    buffer[iB] = dataPos[ iB-N ];
                else
                    buffer[iB] = rData[ rnData-1 ];
            /* __syncthreads() */

            /* handle inner points with enough neighbors on each side */
            #pragma omp parallel for
            for ( unsigned iB = N; iB < rnThreads+N; ++iB )
            if ( &dataPos[iB-N] < &rData[rnData] )
            {
                /* calculate weighted sum */
                T_Prec sum = 0;
                for ( unsigned iW=0, iVal=iB-N; iW < rnWeights; ++iW, ++iVal )
                    sum += buffer[iVal] * rWeights[iW];
                /* write result back into memory (in-place) */
                dataPos[iB-N] = sum;
            }
        }
    }

    template<class T_Prec>
    void gaussianBlur
    (
        T_Prec *     const rData ,
        unsigned int const rnData,
        double       const rSigma
    )
    {
        constexpr int nKernelElements = 64;
        T_Prec pKernel[nKernelElements];
        const int kernelSize = calcGaussianKernel( rSigma, (T_Prec*) pKernel, nKernelElements );
        assert( kernelSize <= nKernelElements );
        applyKernel( rData, rnData, (T_Prec*) pKernel, kernelSize );
    }

    template<class T_Prec>
    void gaussianBlurHorizontal
    (
        T_Prec *     const rData  ,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        double       const rSigma
    )
    {
        const int nKernelElements = 64;
        T_Prec pKernel[64];
        const int kernelSize = calcGaussianKernel( rSigma, (T_Prec*) pKernel, nKernelElements );
        assert( kernelSize <= nKernelElements );


        for ( T_Prec * curRow = rData; curRow < &rData[rnDataX*rnDataY]; curRow += rnDataX )
            applyKernel( curRow, rnDataX, (T_Prec*) pKernel, kernelSize );
    }

    template<class T_Prec>
    void gaussianBlurVerticalUncached
    (
        T_Prec *     const rData  ,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        double       const rSigma
    )
    {
        /* calculate Gaussian kernel */
        const unsigned nKernelElements = 64;
        T_Prec pKernel[64];
        const unsigned kernelSize = calcGaussianKernel( rSigma, (T_Prec*) pKernel, nKernelElements );
        assert( kernelSize <= nKernelElements );
        assert( kernelSize % 2 == 1 );
        const unsigned nKernelHalf = (kernelSize-1)/2;

        /* apply kernel vertical. Make use of cache lines / super words by
         * calculating nColsCacheLine in parallel. For a CUDA device a super
         * word consists of 32 float values = 128 Byte, meaning we can calculate
         * nColsCacheLine = 32 in parallel for T_Prec = float. On the CPU the
         * cache line is 64 Byte which would correspond to AVX512, if it was
         * available, meaning nColsCacheLine = 16
         * @todo: make it work if nColsCacheLine != rnDataX! */
        const unsigned nColsCacheLine = rnDataX;
        /* must be at least kernelSize rows! */
        const unsigned nRowsCacheLine = kernelSize;

        /* allocate cache buffer used for shared memory caching of the results.
         * Therefore needs to be at least kernelSize*nColsCacheLine large, else
         * we would have to write-back the buffer before the weighted sum
         * completed! */
        const unsigned bufferSize = nRowsCacheLine*nColsCacheLine;
        auto buffer = new T_Prec[bufferSize];  /* could be in shared memory or cache */
        //assert( rnDataY <= bufferSize/nColsCacheLine );

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
         *    +------+------+------+------+        +------+------+
         *    | a_20 | a_21 | a_22 | a_22 |        | b_00 | b_01 |
         *    +------+------+------+------+        +------+------+
         *    | a_30 | a_31 | a_32 | a_32 |        | b_10 | b_11 |
         *    +------+------+------+------+        +------+------+
         *    | a_40 | a_41 | a_42 | a_42 |        | b_20 | b_21 |
         *    +------+------+------+------+        +------+------+
         *    | a_50 | a_51 | a_52 | a_52 |
         *    +------+------+------+------+
         *
         *        b_0x = w_-1*a_1x + w_0*a_2x + w_1*a_3x
         *        b_1x = w_-1*a_2x + w_0*a_3x + w_1*a_4x
         *        b_1x = w_-1*a_3x + w_0*a_4x + w_1*a_5x
         *        b_1x = w_-1*a_3x + w_0*a_4x + w_1*a_5x
         * @endverbatim
         * In order to reduce global memory accesses, we can reorder the
         * calculation of b_ij so that we can cache one row of a_ij and
         * basically broadcast ist to b_ij:
         * @verbatim
         *  a) cache a_1x  ->  b_0x += w_-1*a_1x
         *  b) cache a_2x  ->  b_0x += w_0*a_2x, b_1x += w_-1*a_2x
         *  c) cache a_3x  ->  b_0x += w_1*a_3x, b_1x += w_0*a_3x, b_2x += w_-1*a_3x
         *  d) cache a_4x  ->                    b_1x += w_1*a_1x, b_2x += w_0*a_4x
         *  e) cache a_5x  ->                                      b_2x += w_1*a_5x
         * @endverbatim
         * the longer the result buffer is, the more often we can reuse a row
         * of a over the full kernel size, but shared memory is limited ->
         * need to find a good tuning parameter for this.
         * In the case were the row is only used one time, it may be
         * advantageous to not buffer it, thereby saving one access to shared
         * memory (~3 cycles) by directly accessing global memory, but that
         * would make the code less readable, larger (kernel code size also is
         * limited!) and may introduce thread divergence! I don't think it
         * would be better.
         *
         * The buffer size needs at least kernelSize rows. If it's equal to
         * kernel size rows, then in every step one row will be completed
         * calculating, meaning it can be written back.
         * This enables us to use a round-robin like calculation:
         *   - after step c we can write-back b_0x to a_3x, we don't need a_3x
         *     anymore after this step.
         *   - because the write-back needs time the next calculation in d
         *     should write to b_1x. This is the case
         *   - the last operation in d would then be an addition to b_3x == b_0x
         *     (use i % 3)
         *   - we also need to zero the buffer we have written-back. that
         *     shouldn't be done directly after the write-back, because that
         *     takes several hundred cycles. Meaning the zeroing always happens
         *     on the last addition.
         *   - the longer the buffer is the more time we have to wait for the
         *     write-back command to end.
         *   - all this waiting because of global memory access may not be all
         *     that important, because we have warps, which already over-occupy
         *     the GPU to lessen these wait times. So the buffer may not have
         *     to be 100 rows long, to wait ~400 cycles. It may also suffice to
         *     make it only 100/32(warps)~5 large, this is given anyway for most
         *     common kernel sizes! (Gaussian kernel size for sigma=1 is 7)
         *
         * About threading / GPU:
         *   - every column can be calculated in parallel
         *   - the vectorsum/broadcastmultiplication over one row can be
         *     parallelized over the size of the buffer. Every thread would
         **/
        T_Prec * a = rData;
        T_Prec * b = buffer;
        T_Prec * w = pKernel+nKernelHalf; /* now we can use w[-1],... */
        T_Prec cachedRowA[nColsCacheLine];
        /**
         * use extension to calculate the first nKernelHalf rows:
         *   - the first row will have no upper rows as neighbors, meaning
         *     we only add up nKernelHalf*a_0x to b_0x
         *   - 2nd row will use the extension (nKernelHalf-1) times and so on
         *   - actually these partial sums could be precomputed like I did
         *     in newtonCotes
         **/
        /* will contain the antiderivative of pKernel */
        T_Prec pKernelInt[nKernelHalf];
        T_Prec sum = T_Prec(0);
        for ( int iW=-nKernelHalf, iRowA=nKernelHalf-1; iRowA >= 0; ++iW, --iRowA )
        {
            sum += w[iW];
            pKernelInt[iRowA] = sum;
        }
        /* now calculate that part of the buffer that uses the extended values
         * for kernelSize=1 -> nKernelHalf=0 we don't need to do such things,
         * but we can to save one operation for the next loop, so that we have
         * fully used a_0x (for that you would have to use '<=' instead of '<'
         * nKernelHalf, furthermore in above pKernelInt calculation would have
         * to be adjusted, to not include w_0 */
        assert( bufferSize >= nKernelHalf );
        assert( nKernelHalf <= rnDataY );
        assert( bufferSize >= nColsCacheLine*nKernelHalf );

        /* cache the row which we extend over the border */
        memcpy( cachedRowA, a, nColsCacheLine*sizeof(a[0]) );
        for ( unsigned iRowA = 0; iRowA < nKernelHalf; ++iRowA )
        {
            T_Prec * bRow = b + iRowA*nColsCacheLine;
            const auto weight = pKernelInt[iRowA];
            /* scalar * rowvector a_0x, could try to write a class to hide this(!)*/
            for ( unsigned iColA = 0; iColA < nColsCacheLine; ++iColA )
            {
                assert( bRow+iColA < buffer+bufferSize );
                bRow[iColA] = weight * cachedRowA[iColA];
            }
        }

        /* set the rest of the buffer to 0 */
        for ( unsigned iRowB = nKernelHalf; iRowB < nRowsCacheLine; ++iRowB )
        {
            T_Prec * bRow = b + iRowB*nColsCacheLine;
            for ( unsigned iColA = 0; iColA < nColsCacheLine; ++iColA )
            {
                assert( bRow+iColA < buffer+bufferSize );
                bRow[iColA] = 0;
            }
        }

        /* The state now is:
         *   b_0x = a_0x * sum_{i=-nKernelHalf}^{ 0} w_i
         *   b_1x = a_0x * sum_{i=-nKernelHalf}^{-1} w_i
         *   ...
         * The main loop now can begin by broadcasting a_1x weighted to the
         * buffer
         *   b_0x += w_0  * a_0x
         *   b_1x += w_-1 * a_0x
         * Note that iRowB + iW = iRowA.
         * The next loop:
         *   b_0x += w_1  * a_1x
         *   b_1x += w_0  * a_1x
         *   b_2x += w_-1 * a_1x
         * In the 2nd loop, the indexes of the buffer can be calculated
         * from iRowB = iRowA - iW. The problem now is the round-robin.
         * In the next step we would want to begin with b_1x
         * (because b_0x must be written back):
         *   b_1x += w_1  * a_2x
         *   b_2x += w_0  * a_2x
         *   b_0x += w_-1 * a_2x
         * meaning we could still iterate over iW and iRowA and calculate iRowB
         * from those two, if we use a modulo operation:
         *   iRowB = ( iRowA-iW )%( nRowsCacheLine )
         * where in the illustrated case nRowsCacheLine = kernelSize = 3
         */

        /* iterate over the rows of a (the data to convolve) in global memory */
        for ( unsigned iRowA = 0; iRowA < rnDataY; ++iRowA )
        {
            /* cache row of a */
            memcpy( cachedRowA, a+iRowA*rnDataX, nColsCacheLine*sizeof(a[0]) );
            /* add the row weighted with different coefficients to the
             * respective rows in the buffer b. Iterate over the weights / buffer
             * rows */
            for ( int iW = (int) nKernelHalf; iW >= - (int) nKernelHalf; --iW )
            {
                int iRowB = iRowA-iW;
                if ( iRowB < 0 or iRowB >= (int) rnDataY )
                    continue;
                iRowB %= nRowsCacheLine;
                /* calculate index for buffer */
                T_Prec * bRow = b + iRowB*nColsCacheLine;
                const T_Prec weight = w[iW];

                /* do scalar multiply-add vector \vec{b} += w_iW * \vec{a} */
                for ( unsigned iCol = 0; iCol < nColsCacheLine; ++iCol )
                {
                    assert( bRow+iCol < buffer+bufferSize );
                    bRow[iCol] += weight * cachedRowA[iCol];
                }
            }
            /* write the line of the buffer, which completed calculating
             * back to global memory */
            const int iRowAWriteBack = iRowA - nKernelHalf;
            if ( iRowAWriteBack < 0 or iRowAWriteBack >= (int) rnDataY )
                continue;
            else
            {
                const int iRowB = iRowAWriteBack % nRowsCacheLine;
                T_Prec * const rowB = b + iRowB * nColsCacheLine;
                T_Prec * const rowA = a + iRowAWriteBack * rnDataX;
                /* @todo: make it work for rnDataX > nRowsCacheLine */
                assert( nColsCacheLine == rnDataX );

                memcpy( rowA,rowB, nColsCacheLine*sizeof(b[0]) );
                /* could and should be done at a later point, so that we don't
                 * need to wait for the writ-back to finish */
                memset( rowB, 0, nColsCacheLine*sizeof(b[0]) );
            }
        }

        /* cache the the last row which we extend over the border */
        memcpy( cachedRowA, a+rnDataX*(rnDataY-1), nColsCacheLine*sizeof(a[0]) );
        for ( unsigned iRowA = rnDataY; iRowA < rnDataY+nKernelHalf; ++iRowA )
        {
            const int iRowAWriteBack = iRowA - nKernelHalf;
            const int iRowB = iRowAWriteBack % nRowsCacheLine;
            T_Prec * const rowA = a + iRowAWriteBack * rnDataX;
            T_Prec * const rowB = b + iRowB*nColsCacheLine;

            const auto weight = pKernelInt[(nKernelHalf-1)-(iRowA-rnDataY)];
            /* scalar * rowvector a_0x, could try to write a class to hide this(!)*/
            for ( unsigned iColA = 0; iColA < nColsCacheLine; ++iColA )
                rowB[iColA] += weight * cachedRowA[iColA];

            memcpy( rowA,rowB, nColsCacheLine*sizeof(b[0]) );
        }

        delete[] buffer;
    }

    inline int min( const int & a, const int & b )
    {
        return a < b ? a : b; // a <? b GNU C++ extension does the same :S!
    }
    inline int max( const int & a, const int & b )
    {
        return a > b ? a : b;
    }
    inline unsigned min( unsigned int const a, unsigned int const b )
    {
        return a < b ? a : b; // a <? b GNU C++ extension does the same :S!
    }
    inline unsigned max( unsigned int const a, unsigned int const b )
    {
        return a > b ? a : b;
    }


    /**
     * Provides a class for a moving window type 2d cache
     **/
    template<class T_Prec>
    struct MovingWindowCache2D
    {
        T_Prec const * const rData;
        unsigned const & rnDataX;
        unsigned const & rnDataY;

        T_Prec * const buffer; /**< pointer to allocated buffer, will not be allocated on constructor because this class needs to be trivial to work on GPU */
        unsigned const & nRowsBuffer;
        unsigned const & nColsBuffer;

        unsigned const & nThreads;
        unsigned const & nKernelHalf;

        inline T_Prec & operator[]( unsigned i ) const
        {
            return buffer[i];
        }

        /**
         * @param[in] nThreads specifies how many values should be calculatable.
         *            Meaning this sets the numbers of rows we need to cache.
         * @param[in] iCol specifies the start column of rData from which we
         *            are to cache data
         **/
        inline void loadNextColumns( unsigned const & iCol ) const
        {
            assert( iCol < rnDataX );

            /* In the first step initialize the left halo buffer cells which will then be moved to the start to the first  */
            #ifndef NDEBUG
            #if DEBUG_GAUSSIAN_CPP == 1
                /* makes it easier to see if we cache the correct data */
                memset( buffer, 0, nBufferSize*sizeof( buffer[0] ) );
                std::cout << "Copy some initial data to buffer:\n";
            #endif
            #endif

            for ( unsigned iRowBuf = nThreads; iRowBuf < nRowsBuffer; ++iRowBuf )
            {
                /* if rnDataY == 1, then we can only load 2*nKernelHalf+1 rows! */
                if ( iRowBuf-nThreads >= rnDataY + 2*nKernelHalf+1 )
                    break;

                for ( unsigned iColBuf = 0; iColBuf < nColsBuffer; ++iColBuf )
                {
                    if ( iCol+iColBuf >= rnDataX )
                        break;

                    const int signedDataRow = int(iRowBuf-nThreads) - (int)nKernelHalf;
                    /* periodic */
                    //const int tmpRow = signedRow % rnDataY;
                    //const int newRow = tmpRow < 0 ? tmpRow + rnDataY : tmpRow;
                    /* extend */
                    const unsigned newRow = min( rnDataY-1, (unsigned) max( 0, signedDataRow ) );
                        assert( newRow < rnDataY );
                    const unsigned iBuf = iRowBuf*nColsBuffer + iColBuf;
                        assert( iBuf < nRowsBuffer*nColsBuffer );
                    const unsigned iData = newRow*rnDataX + iCol+iColBuf;
                        assert( iData < rnDataX*rnDataY );

                    (*this)[ iBuf ] = rData[ iData ];
                }
            }

            #ifndef NDEBUG
            #if DEBUG_GAUSSIAN_CPP == 1
                for ( unsigned iRowBuf = 0; iRowBuf < nRowsBuffer; ++iRowBuf )
                {
                    std::cout << std::setw(3);
                    std::cout << iRowBuf << ": ";
                    std::cout << std::setw(11);
                    for ( unsigned iColBuf = 0; iColBuf < nColsBuffer; ++iColBuf )
                        std::cout << buffer[ iRowBuf*nColsBuffer + iColBuf ] << " ";
                    std::cout << "\n";
                }
            #endif
            #endif
        }
    };

    template<class T_Prec>
    void gaussianBlurVertical
    (
        T_Prec *     const rData  ,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        double       const rSigma
    )
    {
        /* calculate Gaussian kernel */
        const unsigned nKernelElements = 64;
        T_Prec pKernel[64];
        const unsigned kernelSize = calcGaussianKernel( rSigma, (T_Prec*) pKernel, nKernelElements );
            assert( kernelSize <= nKernelElements );
            assert( kernelSize % 2 == 1 );
        const unsigned nKernelHalf = (kernelSize-1)/2;

        T_Prec * const w = pKernel+nKernelHalf; /* now we can use w[-1],... */

        #ifndef NDEBUG
        #if DEBUG_GAUSSIAN_CPP == 1
            std::cout << "\nConvolve Kernel : \n";
            for ( unsigned iW = 0; iW < kernelSize; ++iW )
                std::cout << pKernel[iW] << ( iW == kernelSize-1 ? "" : " " );
            std::cout << "\n";

            std::cout << "\nInput Matrix to convolve vertically : \n";
            for ( unsigned iRow = 0; iRow < rnDataY; ++iRow )
            {
                std::cout << std::setw(11);
                for ( unsigned iCol = 0; iCol < rnDataX; ++iCol )
                    std::cout << rData[ iRow*rnDataX + iCol ] << " ";
                std::cout << "\n";
            }
        #endif
        #endif

        /* 16*4 Byte (Float) = 64 Byte ~ 1 cache line on sandybridge */
        const unsigned nColsBuffer = 64 / sizeof( rData[0] );
        /* must be at least kernelSize rows! and should fit into L1-Cache of
         * e.g. 32KB */
        const unsigned nRowsBuffer = 30000 / 4 / nColsBuffer;
            assert( nRowsBuffer >= kernelSize );
        const unsigned nThreads = nRowsBuffer - 2*nKernelHalf;
            assert( nThreads > 0 );
        const unsigned nBufferSize = nColsBuffer * nRowsBuffer;
        auto pBuffer = new T_Prec[nBufferSize];

        /* actually C++11, but only implemented in GCC 5 ! */
        //static_assert( std::is_trivially_copyable< MovingWindowCache2D<T_Prec> >::value );
        MovingWindowCache2D<T_Prec> buffer
        {
            rData, rnDataX, rnDataY,
            pBuffer, nRowsBuffer, nColsBuffer,
            nThreads, nKernelHalf
        };

        for ( unsigned iCol = 0; iCol < rnDataX; iCol += nColsBuffer )
        {
            buffer.loadNextColumns( iCol );

            for ( unsigned iRow = 0; iRow < rnDataY; iRow += nRowsBuffer-kernelSize+1 )
            {
                /* move as many rows as we calculated in last iteration with
                 * threads */
                //memcpy( buffer, buffer + nThreads*nColsBuffer, (nBufferSize - nThreads*nColsBuffer) * sizeof( buffer[0] ) );
                for ( unsigned iRowTmp = 0; iRowTmp < nRowsBuffer-nThreads; ++iRowTmp )
                for ( unsigned iColTmp = 0; iColTmp < nColsBuffer; ++iColTmp )
                {
                    unsigned int const iTarget = iRowTmp * nColsBuffer + iColTmp;
                    assert( iTarget < nBufferSize );
                    unsigned int const iSrc
                        = ( iRowTmp + nThreads ) * nColsBuffer + iColTmp;
                    assert( iSrc < nBufferSize );
                    buffer[ iTarget ] = buffer[ iSrc ];
                }

                /* cache new values to freed places to the right  */
                for ( unsigned iRowBuf = nRowsBuffer - nThreads; iRowBuf < nRowsBuffer; ++iRowBuf )
                {
                    if ( iRow+iRowBuf >= rnDataY + 2*nKernelHalf )
                        break;

                    for ( unsigned iColBuf = 0; iColBuf < nColsBuffer; ++iColBuf )
                    {
                        if ( iCol+iColBuf >= rnDataX )
                            break;

                        //std::cout << "Buffer " << iRowBuf << "," << iColBuf << " -> ";
                        /* periodic */
                        //const int signedRow = ( iRow + iW ) % rnDataY;
                        //const int newRow = signedRow < 0 ? signedRow + rnDataY : signedRow;
                        /* extend */
                            assert( rnDataY >= 1 );
                        const unsigned newRow = min( rnDataY-1, (unsigned) max( 0,
                            (int)iRow - (int)nKernelHalf + (int)iRowBuf ) );
                        const unsigned iBuf = iRowBuf*nColsBuffer + iColBuf;
                            assert( iBuf < nBufferSize );
                        const unsigned iData = newRow*rnDataX + iCol+iColBuf;
                            assert( iData < rnDataX*rnDataY );

                        buffer[ iBuf ] = rData[ iData ];

                        //std::cout << newRow << "," << iCol+iColBuf << "\n";
                    }
                }
                #ifndef NDEBUG
                #if DEBUG_GAUSSIAN_CPP == 1
                    std::cout << "Move buffer data to beginning and fill end:\n";
                    for ( unsigned iRowBuf = 0; iRowBuf < nRowsBuffer; ++iRowBuf )
                    {
                        std::cout << std::setw(3);
                        std::cout << iRowBuf << ": ";
                        std::cout << std::setw(11);
                        for ( unsigned iColBuf = 0; iColBuf < nColsBuffer; ++iColBuf )
                            std::cout << buffer[ iRowBuf*nColsBuffer + iColBuf ] << " ";
                        std::cout << "\n";
                    }
                #endif
                #endif

                /* calculate on buffer */
                #ifndef NDEBUG
                #if DEBUG_GAUSSIAN_CPP == 1
                    std::cout << "Calculated values:\n";
                #endif
                #endif
                for ( unsigned iRowBuf = nKernelHalf; iRowBuf < nRowsBuffer-nKernelHalf; ++iRowBuf )
                {
                    if ( iRow + (iRowBuf-nKernelHalf) >= rnDataY )
                        break;

                    #ifndef NDEBUG
                    #if DEBUG_GAUSSIAN_CPP == 1
                        std::cout << std::setw(3) << iRow + (iRowBuf-nKernelHalf) << ": ";
                    #endif
                    #endif

                    for ( unsigned iColBuf = 0; iColBuf < nColsBuffer; ++iColBuf )
                    {
                        if ( iCol+iColBuf >= rnDataX )
                            break;

                        //std::cout << "Buffer " << iRowBuf << "," << iColBuf << "\n";
                        /* calculate weighted sum */
                        T_Prec sum = 0;
                        for ( int iW = -nKernelHalf; iW <= (int)nKernelHalf; ++iW )
                        {
                            assert( (int)iRowBuf + iW >= 0 );
                            const unsigned iBuf = unsigned( iRowBuf+iW ) * nColsBuffer + iColBuf;
                            assert( iBuf < nBufferSize );
                            sum += buffer[iBuf] * w[iW];
                        }
                        assert( iRowBuf >= nKernelHalf );
                        const unsigned iData = ( iRow + (iRowBuf-nKernelHalf) ) * rnDataX + iCol+iColBuf;
                        assert( iData < rnDataX*rnDataY );
                        rData[ iData ] = sum;

                        #ifndef NDEBUG
                        #if DEBUG_GAUSSIAN_CPP == 1
                            std::cout << std::setw(11) << sum << " ";
                        #endif
                        #endif
                    }

                    #ifndef NDEBUG
                    #if DEBUG_GAUSSIAN_CPP == 1
                        std::cout << "\n";
                    #endif
                    #endif
                }
            }
        }

        delete[] pBuffer;
    }


    template<class T_Prec>
    void gaussianBlur
    (
        T_Prec *     const rData  ,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        double       const rSigma
    )
    {
        assert( rData != NULL );
        gaussianBlurHorizontal( rData,rnDataX,rnDataY,rSigma );
        gaussianBlurVertical  ( rData,rnDataX,rnDataY,rSigma );
    }


    template<class T_PREC>
    void gaussianBlurFft
    (
        T_PREC *     const rData  ,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        double       const rSigma
    )
    {
        #ifdef USE_FFTW
            //
        #else
            assert( false && "gaussianBlurFft can only be used if compiled with USE_FFTW CMake option." );
        #endif
    }


} // namespace libs
} // namespace imresh
