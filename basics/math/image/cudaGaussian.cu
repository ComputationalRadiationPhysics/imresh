#include "cudaGaussian.h"


namespace imresh {
namespace math {
namespace image {


#define DEBUG_GAUSSIAN_CPP 0


template<class T_PREC>
void cudaApplyKernel
( T_PREC * const rData, const unsigned rnData,
  const T_PREC * const rWeights, const unsigned rnWeights,
  const unsigned rnThreads )
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
     * this means, the buffer size needs to be rnThreads + 2*N elements long:
     *
     * +--+--+--+--+--+--+--+--+--+--+--+--+
     * |xx|xx|  |  |  |  |  |  |  |  |yy|yy|
     * +--+--+--+--+--+--+--+--+--+--+--+--+
     * <-----><---------------------><----->
     *   N=2       rnThreads = 8      N=2
     *
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
     * in parallel. The elements marked yy are the old elements from the right
     * border, which were only used readingly up till now. The move of the
     * 2*N elements may be preventable by using a modulo address access, but
     * a move in shared memory / cache is much faster than waiting for the
     * rest of the array to be filled with new data from global i.e. uncached
     * memory.
     **/
    const int bufferSize = rnThreads + 2*N;
    T_PREC * buffer = (T_PREC*) malloc( sizeof(T_PREC)*bufferSize );

    /* In the first step initialize the left border to the same values (extend) */
    const T_PREC leftBorderValue = rData[0];
    #pragma omp parallel for
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
            buffer[iB] = dataPos[ std::min( iB-N, rnData-1 ) ];
        /* __syncthreads() */

        /* handle inner points with enough neighbors on each side */
        #pragma omp parallel for
        for ( unsigned iB = N; iB < rnThreads+N; ++iB )
        {
            if ( &dataPos[iB-N] > &rData[rnData-1] )
                continue;
            /* calculate weighted sum */
            T_PREC sum = 0;
            for ( unsigned iW=0, iVal=iB-N; iW < rnWeights; ++iW, ++iVal )
                sum += buffer[iVal] * rWeights[iW];
            /* write result back into memory (in-place) */
            dataPos[iB-N] = sum;
        }
    }
}



template<class T_PREC>
void cudaGaussianBlur
( T_PREC * rData, int rnData, double rSigma )
{
    const int nKernelElements = 64;
    T_PREC pKernel[64];
    const int kernelSize = calcGaussianKernel( rSigma, pKernel, nKernelElements );
    assert( kernelSize <= nKernelElements );
    applyKernel( rData, rnData, pKernel, kernelSize );
}

template<class T_PREC>
void cudaGaussianBlurHorizontal
( T_PREC * rData, int rnDataX, int rnDataY, double rSigma )
{
    const int nKernelElements = 64;
    T_PREC pKernel[64];
    const int kernelSize = calcGaussianKernel( rSigma, pKernel, nKernelElements );
    assert( kernelSize <= nKernelElements );


    for ( T_PREC * curRow = rData; curRow < &rData[rnDataX*rnDataY]; curRow += rnDataX )
        cudaApplyKernel( curRow, rnDataX, pKernel, kernelSize );
}

template<class T_PREC>
void cudaGaussianBlurVertical
( T_PREC * rData, int rnDataX, int rnDataY, double rSigma )
{

    /* calculate Gaussian kernel */
    const int nKernelElements = 64;
    T_PREC pKernel[64];
    const int kernelSize = calcGaussianKernel( rSigma, pKernel, nKernelElements );
    assert( kernelSize <= nKernelElements );
    assert( kernelSize % 2 == 1 );
    const int nKernelHalf = (kernelSize-1)/2;

    /* apply kernel vertical. Make use of cache lines / super words by
     * calculating nColsCacheLine in parallel. For a CUDA device a super
     * word consists of 32 float values = 128 Byte, meaning we can calculate
     * nColsCacheLine = 32 in parallel for T_PREC = float. On the CPU the
     * cache line is 64 Byte which would correspond to AVX512, if it was
     * available, meaning nColsCacheLine = 16
     * @todo: make it work if nColsCacheLine != rnDataX! */
    const int nColsCacheLine = rnDataX;
    /* must be at least kernelSize rows! */
    const int nRowsCacheLine = kernelSize;

    /* allocate cache buffer used for shared memory caching of the results.
     * Therefore needs to be at least kernelSize*nColsCacheLine large, else
     * we would have to write-back the buffer before the weighted sum
     * completed! */
    const int bufferSize = nColsCacheLine*nColsCacheLine;
    T_PREC buffer[bufferSize];  /* could be in shared memory or cache */
    //assert( rnDataY <= bufferSize/nColsCacheLine );

    /**
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
     *
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
     * the longer the result buffer is, the more often we can reuse a row of
     * a over the full kernel size, but shared memory is limited -> need to
     * find a good tuning parameter for this.
     * In the case were the row is only used one time, it may be advantageous
     * to not buffer it, thereby saving one access to shared memory (~3 cycles)
     * by directly accessing global memory, but that would make the code
     * less readable, larger (kernel code size also is limited!) and may
     * introduce thread divergence! I don't think it would be better.
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
     *   - we also need to zero the buffer we have written-back. that shouldn't
     *     be done directly after the write-back, because that takes several
     *     hundred cycles. Meaning the zeroing always happens on the last
     *     addition.
     *   - the longer the buffer is the more time we have to wait for the
     *     write-back command to end.
     *   - all this waiting because of global memory access may not be all that
     *     important, because we have warps, which already over-occupy the
     *     GPU to lessen these wait times. So the buffer may not have to be
     *     100 rows long, to wait ~400 cycles. It may also suffice to make it
     *     only 100/32(warps)~5 large, this is given anyway for most common
     *     kernel sizes! (Gaussian kernel size for sigma=1 is 7)
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
     * for kernelSize=1 -> nKernelHalf=0 we don't need to do such things, but
     * we can to save one operation for the next loop, so that we have fully
     * used a_0x (for that you would have to use '<=' instead of '<'
     * nKernelHalf, furthermore in above pKernelInt calculation would have
     * to be adjusted, to not include w_0 */
    assert( bufferSize >= nKernelHalf );
    assert( nKernelHalf <= rnDataY );
    assert( bufferSize >= nColsCacheLine*nKernelHalf );

    /* cache the row which we extend over the border */
    memcpy( cachedRowA, a, nColsCacheLine*sizeof(a[0]) );
    for ( int iRowA = 0; iRowA < nKernelHalf; ++iRowA )
    {
        T_PREC * bRow = b + iRowA*nColsCacheLine;
        const auto weight = pKernelInt[iRowA];
        /* scalar * rowvector a_0x, could try to write a class to hide this(!)*/
        for ( int iColA = 0; iColA < nColsCacheLine; ++iColA )
            bRow[iColA] = weight * cachedRowA[iColA];
    }

    /* set the rest of the buffer to 0 */
    for ( int iRowB = nKernelHalf; iRowB < nRowsCacheLine; ++iRowB )
    {
        T_PREC * bRow = b + iRowB*nColsCacheLine;
        for ( int iColA = 0; iColA < nColsCacheLine; ++iColA )
            bRow[iColA] = 0;
    }

    /* The state now is:
     *   b_0x = a_0x * sum_{i=-nKernelHalf}^{ 0} w_i
     *   b_1x = a_0x * sum_{i=-nKernelHalf}^{-1} w_i
     *   ...
     * The main loop now can begin by broadcasting a_1x weighted to the buffer
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

#   if DEBUG_GAUSSIAN_CPP == 1
    std::cout
    << "Vertical Gaussian, Paramters:\n"
    << " . kernelSize     = " << kernelSize     << "\n"
    << " . nKernelHalf    = " << nKernelHalf    << "\n"
    << " . rnDataX        = " << rnDataX        << "\n"
    << " . rnDataY        = " << rnDataY        << "\n"
    << " . nRowsCacheLine = " << nRowsCacheLine << "\n"
    << " . nColsCacheLine = " << nColsCacheLine << "\n"
    << " . pKernelInt     = {";
    for ( int i=0; i <= nKernelHalf; ++i ) std::cout << pKernelInt[i] << " ";
    std::cout << "}\n";
    std::cout << " . pKernel        = {";
    for ( int i=0; i < kernelSize; ++i ) std::cout << pKernel[i] << " ";
    std::cout << "}\n";
    std::cout
    << " . a              = " << (void*)a << "\n"
    << " . b              = " << (void*)b << "\n";
#   endif

    /* iterate over the rows of a (the data to convolve) in global memory */
    for ( int iRowA = 0; iRowA < rnDataY; ++iRowA )
    {
#       if DEBUG_GAUSSIAN_CPP == 1
            std::cout << "Loop over a_"<<iRowA<<"x:\n";
#       endif
        /* cache row of a */
        memcpy( cachedRowA, a+iRowA*rnDataX, nColsCacheLine*sizeof(a[0]) );
        /* add the row weighted with different coefficients to the
         * respective rows in the buffer b. Iterate over the weights / buffer
         * rows */
        for ( int iW = nKernelHalf; iW >= -nKernelHalf; --iW )
        {
            int iRowB = iRowA-iW;
            if ( iRowB < 0 or iRowB >= rnDataY )
                continue;
            iRowB %= nRowsCacheLine;
            /* calculate index for buffer */
            T_PREC * bRow = b + iRowB*nColsCacheLine;
            const T_PREC weight = w[iW];

#           if DEBUG_GAUSSIAN_CPP == 1
                std::cout << "  b_"<<iRowB<<"x += w_"<<iW<<" * a_"<<iRowA<<"x (w="<<w[iW]<<")\n";
#           endif

            /* do scalar multiply-add vector \vec{b} += w_iW * \vec{a} */
            for ( int iCol = 0; iCol < nColsCacheLine; ++iCol )
                bRow[iCol] += weight * cachedRowA[iCol];
        }
        /* write the line of the buffer, which completed calculating
         * back to global memory */
        const int iRowAWriteBack = iRowA - nKernelHalf;
        if ( iRowAWriteBack < 0 or iRowAWriteBack >= rnDataY )
            continue;
        else
        {
            const int iRowB = iRowAWriteBack % nRowsCacheLine;
            T_PREC * const rowB = b + iRowB * nColsCacheLine;
            T_PREC * const rowA = a + iRowAWriteBack * rnDataX;
            /* @todo: make it work for rnDataX > nRowsCacheLine */
            assert( nColsCacheLine == rnDataX );

#           if DEBUG_GAUSSIAN_CPP == 1
                std::cout << "Write back b_"<<iRowB<<"x to a_"<<iRowAWriteBack<<"x"
                          << ", i.e. "<<(void*)rowB<<" -> "<<(void*)rowA<<"\n";
#           endif

            memcpy( rowA,rowB, nColsCacheLine*sizeof(b[0]) );
            /* could and should be done at a later point, so that we don't
             * need to wait for the writ-back to finish */
            memset( rowB, 0, nColsCacheLine*sizeof(b[0]) );
        }
    }

    /* cache the the last row which we extend over the border */
    memcpy( cachedRowA, a+rnDataX*(rnDataY-1), nColsCacheLine*sizeof(a[0]) );
    for ( int iRowA = rnDataY; iRowA < rnDataY+nKernelHalf; ++iRowA )
    {
        const int iRowAWriteBack = iRowA - nKernelHalf;
        const int iRowB = iRowAWriteBack % nRowsCacheLine;
        T_PREC * const rowA = a + iRowAWriteBack * rnDataX;
        T_PREC * const rowB = b + iRowB*nColsCacheLine;

        const auto weight = pKernelInt[(nKernelHalf-1)-(iRowA-rnDataY)];
        /* scalar * rowvector a_0x, could try to write a class to hide this(!)*/
        for ( int iColA = 0; iColA < nColsCacheLine; ++iColA )
            rowB[iColA] += weight * cachedRowA[iColA];

        memcpy( rowA,rowB, nColsCacheLine*sizeof(b[0]) );
    }

}

template<class T_PREC>
void cudaGaussianBlur
( T_PREC * rData, int rnDataX, int rnDataY, double rSigma )
{
    cudaGaussianBlurHorizontal( rData,rnDataX,rnDataY,rSigma );
    cudaGaussianBlurVertical  ( rData,rnDataX,rnDataY,rSigma );
}



/* Explicitely instantiate certain template arguments to make an object file */
template void cudaApplyKernel<float >( float  * const rData, const unsigned rnData, const float  * const rWeights, const unsigned rnWeights, const unsigned rnThreads );
template void cudaApplyKernel<double>( double * const rData, const unsigned rnData, const double * const rWeights, const unsigned rnWeights, const unsigned rnThreads );

template void cudaGaussianBlur<float >( float  * rData, int rnData, double rSigma );
template void cudaGaussianBlur<double>( double * rData, int rnData, double rSigma );
template void cudaGaussianBlur<float >( float  * rData, int rnDataX, int rnDataY, double rSigma );
template void cudaGaussianBlur<double>( double * rData, int rnDataX, int rnDataY, double rSigma );
template void cudaGaussianBlurHorizontal<float >( float  * rData, int rnDataX, int rnDataY, double rSigma );
template void cudaGaussianBlurHorizontal<double>( double * rData, int rnDataX, int rnDataY, double rSigma );
template void cudaGaussianBlurVertical<float >( float  * rData, int rnDataX, int rnDataY, double rSigma );
template void cudaGaussianBlurVertical<double>( double * rData, int rnDataX, int rnDataY, double rSigma );


} // namespace image
} // namespace math
} // namespace imresh
