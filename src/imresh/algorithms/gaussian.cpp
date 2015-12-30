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


#include "gaussian.h"


namespace imresh
{
namespace algorithms
{


    #define DEBUG_GAUSSIAN_CPP 0


    template<class T_PREC>
    void applyKernel
    (
        T_PREC * const & rData,
        const unsigned & rnData,
        const T_PREC * const & rWeights,
        const unsigned & rnWeights,
        const unsigned & rnThreads
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
        T_PREC * buffer = (T_PREC*) malloc( sizeof(T_PREC)*bufferSize );

        /* In the first step initialize the left border to the same values (extend) */
        const T_PREC leftBorderValue = rData[0];
        for ( unsigned iB = 0; iB < N; ++iB )
            buffer[ bufferSize-2*N+iB ] = leftBorderValue;

        /* Loop over buffers. If rnData == rnThreads then the buffer will
         * exactly suffice, meaning the loop will only be run 1 time */
        for ( T_PREC * dataPos = rData; dataPos < &rData[rnData]; dataPos += rnThreads )
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
                T_PREC sum = 0;
                for ( unsigned iW=0, iVal=iB-N; iW < rnWeights; ++iW, ++iVal )
                    sum += buffer[iVal] * rWeights[iW];
                /* write result back into memory (in-place) */
                dataPos[iB-N] = sum;
            }
        }
    }


    /**
     * We need to choose a size depending on the sigma, for which the kernel
     * will still be 100% accurate, even though we cut of quite a bit.
     * For 8bit images the channels can only store 0 to 255. The most extreme
     * case, where the farest neighbors could still influence the current pixel
     * would be all the pixels inside the kernel being 0 and all the others
     * being 255:
     * @verbatim
     *      255  -----+       +-------
     *                |       |
     *        0       |_______|
     *                    ^
     *             pixel to calculate
     *                <------->
     *      kernel size = 7, i.e. Nw = 3
     *  (number of neighbors in each directions)
     * @endverbatim
     * The only way the outer can influence the interior would be, if the
     * weighted sum over all was > 0.5. Meaning:
     * @f[
     * 255 \int\limits_{-\infty}^{x_\mathrm{cutoff}} \frac{1}{\sqrt{2\pi}\sigma}
     * e^{-\frac{x^2}{2\sigma^2} \mathrm{d}x = 255 \frac{1}{2}
     * \mathrm{erfc}\left( -\frac{ x_\mathrm{cutoff} }{ \sqrt{2}\sigma } \right)
     * \overset{!}{=} 0.5
     * \Rightarrow x_\mathrm{cutoff} = -\sqrt{2}\sigma
     *   \erfc^{-1}\left( \frac{1}{2 \cdot 255} \right)
     *   = -2.884402748387961466 \sigma
     * @f]
     * This result means, that for @f[ \sigma=1 @f] the kernel size should
     * be 3 to the left and 3 to the right, meaning 7 weights large.
     * The center pixel, which we want to update goes is in the range [-0.5,0.5]
     * The neighbor pixel in [-1.5,-0.5], then [-2.5,-1.5]. So we are very
     * very close to the 2.88440, but we should nevertheless include the
     * pixel at [-3.5,-2.5] to be correct.
     **/
    template<class T_PREC>
    int calcGaussianKernel
    (
        const double & rSigma,
        T_PREC * const & rWeights,
        const unsigned & rnWeights,
        const double & rMinAbsoluteError
    )
    {
    /* @todo: inverfc, e.g. with minimax (port python version to C/C++)
     *        the inverse erfc diverges at 0, this makes it hard to find a
     *        a polynomial approximation there, but maybe I could rewrite
     *        minimax algorithm to work with \sum a_n/x**n
     *        Anyway, the divergence is also bad for the kernel Size. In order
     *        to reach floating point single precision of 1e-7 absolute error
     *        the kernel size would be: 3.854659 ok, it diverges much slower
     *        than I though */
        //const int nNeighbors = ceil( erfcinv( 2.0*rMinAbsoluteError ) - 0.5 );
        assert( rSigma >= 0 );
        const int nNeighbors = ceil( 2.884402748387961466 * rSigma - 0.5 );
        const int nWeights   = 2*nNeighbors + 1;
        assert( nWeights > 0 );
        if ( (unsigned) nWeights > rnWeights )
            return nWeights;

        double sumWeightings = 0;
        /* Calculate the weightings. I'm not sure, if this is correct.
         * I mean it could be, that the weights are the integrated gaussian
         * values over the pixel interval, but I guess that would force
         * no interpolation. Depending on the interpolation it wouldn't even
         * be pixel value independent anymore, making this useless, so I guess
         * the normal distribution evaluated at -1,0,1 for a kernel size of 3
         * should be correct ??? */
        const double a =  1.0/( sqrt(2.0*M_PI)*rSigma );
        const double b = -1.0/( 2.0*rSigma*rSigma );
        for ( int i = -nNeighbors; i <= nNeighbors; ++i )
        {
            const T_PREC weight = T_PREC( a*exp( i*i*b ) );
            rWeights[nNeighbors+i] = weight;
            sumWeightings += weight;
        }

        /* scale up or down the kernel, so that the sum of the weights will be 1 */
        for ( int i = -nNeighbors; i <= nNeighbors; ++i )
            rWeights[nNeighbors+i] /= sumWeightings;

        return nWeights;
    }

    template<class T_PREC>
    void gaussianBlur
    (
        T_PREC * const & rData,
        const unsigned & rnData,
        const double & rSigma
    )
    {
        const int nKernelElements = 64;
        T_PREC pKernel[64];
        const int kernelSize = calcGaussianKernel( rSigma, (T_PREC*) pKernel, nKernelElements );
        assert( kernelSize <= nKernelElements );
        applyKernel( rData, rnData, (T_PREC*) pKernel, kernelSize );
    }

    template<class T_PREC>
    void gaussianBlurHorizontal
    (
        T_PREC * const & rData,
        const unsigned & rnDataX,
        const unsigned & rnDataY,
        const double & rSigma
    )
    {
        const int nKernelElements = 64;
        T_PREC pKernel[64];
        const int kernelSize = calcGaussianKernel( rSigma, (T_PREC*) pKernel, nKernelElements );
        assert( kernelSize <= nKernelElements );


        for ( T_PREC * curRow = rData; curRow < &rData[rnDataX*rnDataY]; curRow += rnDataX )
            applyKernel( curRow, rnDataX, (T_PREC*) pKernel, kernelSize );
    }

    template<class T_PREC>
    void gaussianBlurVertical
    (
        T_PREC * const & rData,
        const unsigned & rnDataX,
        const unsigned & rnDataY,
        const double & rSigma
    )
    {
        /* calculate Gaussian kernel */
        const unsigned nKernelElements = 64;
        T_PREC pKernel[64];
        const unsigned kernelSize = calcGaussianKernel( rSigma, (T_PREC*) pKernel, nKernelElements );
        assert( kernelSize <= nKernelElements );
        assert( kernelSize % 2 == 1 );
        const unsigned nKernelHalf = (kernelSize-1)/2;

        /* apply kernel vertical. Make use of cache lines / super words by
         * calculating nColsCacheLine in parallel. For a CUDA device a super
         * word consists of 32 float values = 128 Byte, meaning we can calculate
         * nColsCacheLine = 32 in parallel for T_PREC = float. On the CPU the
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
        T_PREC buffer[bufferSize];  /* could be in shared memory or cache */
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
        T_PREC * a = rData;
        T_PREC * b = buffer;
        T_PREC * w = pKernel+nKernelHalf; /* now we can use w[-1],... */
        T_PREC cachedRowA[nColsCacheLine];
        /**
         * use extension to calculate the first nKernelHalf rows:
         *   - the first row will have no upper rows as neighbors, meaning
         *     we only add up nKernelHalf*a_0x to b_0x
         *   - 2nd row will use the extension (nKernelHalf-1) times and so on
         *   - actually these partial sums could be precomputed like I did
         *     in newtonCotes
         **/
        /* will contain the antiderivative of pKernel */
        T_PREC pKernelInt[nKernelHalf];
        T_PREC sum = T_PREC(0);
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
            T_PREC * bRow = b + iRowA*nColsCacheLine;
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
            T_PREC * bRow = b + iRowB*nColsCacheLine;
            for ( unsigned iColA = 0; iColA < nColsCacheLine; ++iColA )
                bRow[iColA] = 0;
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
                T_PREC * bRow = b + iRowB*nColsCacheLine;
                const T_PREC weight = w[iW];

                /* do scalar multiply-add vector \vec{b} += w_iW * \vec{a} */
                for ( unsigned iCol = 0; iCol < nColsCacheLine; ++iCol )
                    bRow[iCol] += weight * cachedRowA[iCol];
            }
            /* write the line of the buffer, which completed calculating
             * back to global memory */
            const int iRowAWriteBack = iRowA - nKernelHalf;
            if ( iRowAWriteBack < 0 or iRowAWriteBack >= (int) rnDataY )
                continue;
            else
            {
                const int iRowB = iRowAWriteBack % nRowsCacheLine;
                T_PREC * const rowB = b + iRowB * nColsCacheLine;
                T_PREC * const rowA = a + iRowAWriteBack * rnDataX;
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
            T_PREC * const rowA = a + iRowAWriteBack * rnDataX;
            T_PREC * const rowB = b + iRowB*nColsCacheLine;

            const auto weight = pKernelInt[(nKernelHalf-1)-(iRowA-rnDataY)];
            /* scalar * rowvector a_0x, could try to write a class to hide this(!)*/
            for ( unsigned iColA = 0; iColA < nColsCacheLine; ++iColA )
                rowB[iColA] += weight * cachedRowA[iColA];

            memcpy( rowA,rowB, nColsCacheLine*sizeof(b[0]) );
        }

    }

    template<class T_PREC>
    void gaussianBlur
    (
        T_PREC * const & rData,
        const unsigned & rnDataX,
        const unsigned & rnDataY,
        const double & rSigma
    )
    {
        gaussianBlurHorizontal( rData,rnDataX,rnDataY,rSigma );
        gaussianBlurVertical  ( rData,rnDataX,rnDataY,rSigma );
    }

#if false
    template<class T_PREC>
    void gaussianBlur
    (
        T_PREC * const & rData,
        const std::vector<unsigned> & rnData,
        const double & rSigma
    )
    {
        unsigned nElements = 1;
        for ( const auto & dim : rnData )
            nElements *= dim;

        /* would need strideX and maybe strideY (?) for this to work with
         * multi dimensions */
        gaussianBlurHorizontal( rData, nElements / rnData[0],
                                       nElements / rnData[0], rnDataY,rSigma );
        unsigned nDataX = 1;
        for ( int iDim = rnData.size(); iDim >= 0; --iDim )
        {

        }
        gaussianBlurVertical  ( rData,rnDataX,rnDataY,rSigma );
    }
#endif

    /* Explicitely instantiate certain template arguments to make an object
     * file. Furthermore this saves space, as we don't need to write out the
     * data types of all functions to instantiate */
    template< class T >
    void __instantiateAllGaussian( T * arg )
    {
        applyKernel           <T>( NULL, 0, NULL, 0, 0 );
        gaussianBlur          <T>( NULL, 0, 0 );
        gaussianBlur          <T>( NULL, 0, 0, 0.0 );
        //gaussianBlur          <T>( NULL, std::vector<unsigned>(3), 0.0 );
        gaussianBlurHorizontal<T>( NULL, 0, 0, 0.0 );
        gaussianBlurVertical  <T>( NULL, 0, 0, 0.0 );
    }
    template void __instantiateAllGaussian<float >( float  * );
    template void __instantiateAllGaussian<double>( double * );


} // namespace algorithms
} // namespace imresh
