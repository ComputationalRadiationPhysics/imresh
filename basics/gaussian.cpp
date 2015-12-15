#include "gaussian.h"

template<class T_PREC>
void applyKernel
( T_PREC * rData, const int rnData,
  const T_PREC * rWeights, const int rnWeights )
{
    assert( rnWeights % 2 == 1 );
    const int N = (rnWeights-1)/2;

    const int bufferSize = 256;
    T_PREC buffer[bufferSize];  /* could be in shared memory or cache */
    assert( rnData <= bufferSize );

    /* handle left edge. if N==0, then kernel is of size 1, meaning a simple
     * element wise identity or scaling, which doesn't need to handle edeges.
     * If N==1, then there is only the upper left pixel to handle which misses
     * one neighbor.
     * If N==2, then we need to handle 2 pixels, missing 1 and 2 neighbors
     * respectively
     */
    for ( int i = 0; i < N; ++i )
    {
        buffer[i] = 0;
        /* i=0 needs to handle N missing neighbors */
        for ( int k = 0; k < N-i; ++k )
            buffer[i] += rData[0] * rWeights[k];
        /* Now handle those kernel parts, where the neighboring pixels exist.
         * For the last case: i=N-1 this will be */
        for ( int k = N-i; k < rnWeights; ++k )
            buffer[i] += rData[i+(k-N)] * rWeights[k];
    }

    /* Handle inner points with enough neighbors on each side */
    for ( int i = N; i < rnData-N; ++i )
    {
        buffer[i] = 0;
        for ( int iWeight=0, iData=i-N; iWeight < rnWeights; ++iWeight, ++iData )
            buffer[i] += rData[iData] * rWeights[iWeight];
    }

    /* handle right edge */
    for ( int i = rnData-N; i < rnData; ++i )
    {
        buffer[i]= 0;
        /* weight neighbors which exist */
        int iWeight=0, iData=i-N;
        for ( ; iData < rnData; ++iWeight, ++iData )
            buffer[i] += rData[iData] * rWeights[iWeight];
        /* handle missing neighbors */
        for ( ; iWeight < rnWeights; ++iWeight )
            buffer[i] += rData[rnData-1] * rWeights[iWeight];
    }

    memcpy( rData, buffer, rnData*sizeof(T_PREC) );
}


template<class T_PREC>
int calcGaussianKernel
( double rSigma, T_PREC * rWeights, const int rnWeights )
{
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
     *   \erfc^{-1}\left( \frac{1}{255} \right) = -2.884402748387961466 \sigma
     * @f]
     * This result means, that for @f[ \sigma=1 @f] the kernel size should
     * be 3 to the left and 3 to the right, meaning 7 weights large.
     * The center pixel, which we want to update goes is in the range [-0.5,0.5]
     * The neighbor pixel in [-1.5,-0.5], then [-2.5,-1.5]. So we are very
     * very close to the 2.88440, but we should nevertheless include the
     * pixel at [-3.5,-2.5] to be correct.
     **/
    const int nNeighbors = ceil( 2.884402748387961466 * rSigma - 0.5 );
    const int nWeights   = 2*nNeighbors + 1;
    assert( nWeights > 0 );
    if ( nWeights > rnWeights )
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
( T_PREC * rData, int rnData, double rSigma )
{
    const int nKernelElements = 64;
    T_PREC pKernel[64];
    const int kernelSize = calcGaussianKernel( rSigma, pKernel, nKernelElements );
    assert( kernelSize <= nKernelElements );
    applyKernel( rData, rnData, pKernel, kernelSize );
}

template<class T_PREC>
void gaussianBlurHorizontal
( T_PREC * rData, int rnDataX, int rnDataY, double rSigma )
{
    const int nKernelElements = 64;
    T_PREC pKernel[64];
    const int kernelSize = calcGaussianKernel( rSigma, pKernel, nKernelElements );
    assert( kernelSize <= nKernelElements );

    T_PREC * curRow = rData;
    for ( int iRow = 0; iRow < rnDataY; ++iRow, curRow += rnDataX )
        applyKernel( curRow, rnDataX, pKernel, kernelSize );
}

template<class T_PREC>
void gaussianBlurVertical
( T_PREC * rData, int rnDataX, int rnDataY, double rSigma )
{
    const int nKernelElements = 64;
    T_PREC pKernel[64];
    const int kernelSize = calcGaussianKernel( rSigma, pKernel, nKernelElements );
    assert( kernelSize <= nKernelElements );

    /* apply kernel vertical. Make use of cache lines / super words by
     * calculating nRowsCacheLine in parallel. For a CUDA device a superword
     * consists of 32 float values = 128 Byte, meaning we can calculate
     * nRowsCacheLine = 32 in parallel for T_PREC = float. On the CPU the
     * cache line is 64 Byte which would correspond to AVX512, if it was
     * available, meaning nRowsCacheLine = 16 */
    const int nRowsCacheLine = 32;

    assert( kernelSize % 2 == 1 );
    const int nKernelHalf = (kernelSize-1)/2;

    const int bufferSize = 256*nRowsCacheLine;
    T_PREC buffer[bufferSize];  /* could be in shared memory or cache */
    assert( rnDataY <= bufferSize/nRowsCacheLine );

#if 1==0
    /* go through all columns summing over them with the kernel */
    T_PREC curCol = rData;
    for ( int iCol = 0; iCol < rnDataX; ++iCol, curCol += nRowsCacheLine )
    {
        /* the buffer will contain the weighted sums */
        memset( buffer,0,bufferSize*sizeof(buffer[0]) );

        /* sum up over column */
        T_PREC * curRow = curCol;
        for ( int iRow = 0; iRow < rnDataY; ++iRow, curRow += rnDataX )
        {
            /* do this for all columns in buffer, except for the last case,
             * where rnDataX % nRowsCacheLine may uneqal than 0 */
            for ( int iBufferCol = 0; iBufferCol < nRowsCacheLine; ++iBufferCol )
            {
                buffer[iBufferCol] += rData[iData] * rWeights[iWeight];
            }
        }
    }
#endif
}


template<class T_PREC>
void gaussianBlur
( T_PREC * rData, int rnDataX, int rnDataY, double rSigma )
{
    gaussianBlurHorizontal( rData,rnDataX,rnDataY,rSigma );
    gaussianBlurVertical  ( rData,rnDataX,rnDataY,rSigma );
}





/* Explicitely instantiate certain template arguments to make an object file */

template void applyKernel<float>
( float * rData, const int rnData,
  const float * rWeights, const int rnWeights );
template void applyKernel<double>
( double * rData, const int rnData,
  const double * rWeights, const int rnWeights );

template void gaussianBlur<float >( float  * rData, int rnData, double rSigma );
template void gaussianBlur<double>( double * rData, int rnData, double rSigma );
template void gaussianBlur<float >( float  * rData, int rnDataX, int rnDataY, double rSigma );
template void gaussianBlur<double>( double * rData, int rnDataX, int rnDataY, double rSigma );
template void gaussianBlurHorizontal<float >( float  * rData, int rnDataX, int rnDataY, double rSigma );
template void gaussianBlurHorizontal<double>( double * rData, int rnDataX, int rnDataY, double rSigma );
template void gaussianBlurVertical<float >( float  * rData, int rnDataX, int rnDataY, double rSigma );
template void gaussianBlurVertical<double>( double * rData, int rnDataX, int rnDataY, double rSigma );
