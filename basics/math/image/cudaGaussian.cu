
#include "cudaGaussian.h"
#include "cudacommon.h"

namespace imresh {
namespace math {
namespace image {


#define DEBUG_GAUSSIAN_CPP 0


template<class T>
__device__ inline T * ptrMin ( T * const a, T * const b )
{
    return a < b ? a : b;
}

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
 **/
template<class T_PREC>
__global__ void cudaKernelApplyKernel
(
  T_PREC * const rdpData, const unsigned rImageWidth,
  T_PREC const * const rdpWeights, const unsigned N
)
{
    assert( N > 0 );
    assert( blockDim.y == 1 and blockDim.z == 1 );
    assert(  gridDim.y == 1 and  gridDim.z == 1 );

    /* If more than 1 block, then each block works on a separate line.
     * Each line borders will be extended. So mutliple blocks can't be used
     * to blur one very very long line even faster! */
    const int & nThreads = blockDim.x;
    T_PREC * const data = &rdpData[ blockIdx.x * rImageWidth ];

    /* manage dynamically allocated shared memory */
    extern __shared__ T_PREC smBlock[];
    const int nWeights = 2*N+1;
    const int bufferSize = nThreads + 2*N;
    T_PREC * const smWeights = smBlock;
    T_PREC * const smBuffer  = &smBlock[ nWeights ];
    __syncthreads();
    /* cache the weights to shared memory @todo: more efficient possible ??? */
    if ( threadIdx.x == 0 )
        memcpy( smWeights, rdpWeights, sizeof(T_PREC)*nWeights );

    /* In the first step initialize the left border to the same values (extend)
     * It's problematic to use threads for this for loop, because it is not
     * guaranteed, that blockDim.x >= N */
    const T_PREC leftBorderValue = data[0];
    if ( threadIdx.x == 0 )
        for ( unsigned iB = 0; iB < N; ++iB )
            smBuffer[ bufferSize-2*N+iB ] = leftBorderValue;

    /* Loop over buffers. If rnData == rnThreads then the buffer will
     * exactly suffice, meaning the loop will only be run 1 time.
     * The for loop break condition is the same for all threads, so it is
     * safe to use __syncthreads() inside */
    for ( T_PREC * curDataRow = data; curDataRow < &data[rImageWidth]; curDataRow += nThreads )
    {
        /* move last N elements to the front of the buffer */
        __syncthreads();
        if ( threadIdx.x == 0 )
            memcpy( smBuffer, &smBuffer[ bufferSize-2*N ], N*sizeof(T_PREC) );

        /* Load rnThreads+N data elements into buffer. If data end reached,
         * fill buffer with last data element */
        __syncthreads();
        const unsigned iBuf = N + threadIdx.x;
        T_PREC * const datum = ptrMin( &curDataRow[ threadIdx.x ],
                                       &data[ rImageWidth-1 ] );
        smBuffer[ iBuf ] = *datum;
        /* again this is hard to parallelize if we don't have as many threads
         * as the kernel is wide. Furthermore we can't use memcpy, because
         * it may be, that we need to extend a value, because we reached the
         * border */
        if ( threadIdx.x == 0 )
        {
            for ( unsigned iB = N+nThreads; iB < nThreads+2*N; ++iB )
            {
                T_PREC * const datum = ptrMin( &curDataRow[ iB-N ],
                                               &data[ rImageWidth-1 ] );
                smBuffer[iB] = *datum;
            }
        }
        __syncthreads();

        /* calculated weighted sum on inner points in buffer, but only if
         * the value we are at is actually needed: */
        if ( &curDataRow[iBuf-N] < &data[rImageWidth] )
        {
            T_PREC sum = T_PREC(0);
            /* this for loop is done by each thread and should for large
             * enough kernel sizes sufficiently utilize raw computing power */
            for ( T_PREC * w = smWeights, * x = &smBuffer[iBuf-N];
                  w < &smWeights[nWeights]; ++w, ++x )
                sum += (*w) * (*x);
            /* write result back into memory (in-place). No need to wait for
             * all threads to finish, because we write into global memory, to
             * values we already buffered into shared memory! */
            curDataRow[iBuf-N] = sum;
        }
    }
}

template<class T_PREC>
void cudaApplyKernel
( T_PREC * const rdpData, const unsigned rnData,
  const T_PREC * const rdpWeights, const unsigned rnWeights,
  const unsigned rnThreads )
{
    assert( rnWeights > 0 );
    assert( rnWeights % 2 == 1 );
    assert( rnThreads > 0 );

    const unsigned N = (rnWeights-1)/2;
    const unsigned bufferSize = rnThreads + 2*N;

    cudaKernelApplyKernel<<<
        1,rnThreads,
        sizeof(T_PREC)*( rnWeights + bufferSize )
    >>>( rdpData, rnData, rdpWeights, N );
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
( T_PREC * rdpData, int rnDataX, int rnDataY, double rSigma )
{
    const int nKernelElements = 64;
    T_PREC pKernel[64];
    const int kernelSize = calcGaussianKernel( rSigma, pKernel, nKernelElements );
    assert( kernelSize <= nKernelElements );

    /* upload kernel to GPU */
    T_PREC * dpKernel;
    CUDA_ERROR( cudaMalloc( &dpKernel, sizeof(T_PREC)*kernelSize ) );
    CUDA_ERROR( cudaMemcpy(  dpKernel, pKernel, sizeof(T_PREC)*kernelSize, cudaMemcpyHostToDevice ) );

    /* the image must be at least nThreads threads wide, else many threads
     * will only sleep. The number of blocks is equal to the image height.
     * Every block works on 1 image line. The number of Threads is limited
     * by the hardware to be e.g. 512 or 1024. The reason for this is the
     * limited shared memory size! */
    const unsigned nThreads = 256;
    const unsigned nBlocks  = rnDataY;
    const unsigned N = (kernelSize-1)/2;
    const unsigned bufferSize = nThreads + 2*N;

    cudaKernelApplyKernel<<<
        nBlocks,nThreads,
        sizeof(T_PREC)*( kernelSize + bufferSize )
    >>>( rdpData, rnDataX, dpKernel, N );
    CUDA_ERROR( cudaDeviceSynchronize() );

    CUDA_ERROR( cudaFree( dpKernel ) );
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
 **/
template<class T_PREC>
__global__ void cudaKernelApplyKernelVertically
(
  T_PREC * const rdpData, const unsigned rnDataX, const unsigned rnDataY,
  T_PREC const * const rdpWeights, const unsigned N
)
{
    assert( N > 0 );
    assert( blockDim.z == 1 );
    assert( gridDim.y == 1 and  gridDim.z == 1 );

    /* the shared memory buffer dimensions */
    const unsigned nColsCacheLine = blockDim.x;
    const unsigned nRowsCacheLine = blockDim.y + 2*N;

    /* Each block works on a separate group of columns */
    T_PREC * const data = &rdpData[ blockIdx.x * blockDim.x ];
    /* the rightmost block might not be full. In that case we need to mask
     * those threads working on the columns right of the image border */
    const bool iAmSleeping = blockIdx.x * blockDim.x + threadIdx.x >= rnDataX;

    /* The dynamically allocated shared memory buffer will fit the weights and
     * the values to calculate + the 2*N neighbors needed to calculate them */
    extern __shared__ T_PREC smBlock[];
    const unsigned nWeights   = 2*N+1;
    const unsigned bufferSize = nColsCacheLine * nRowsCacheLine;
    T_PREC * const smWeights  = smBlock;
    T_PREC * const smBuffer  = &smBlock[ nWeights ];
    __syncthreads();

    /* cache the weights to shared memory */
    if ( threadIdx.x == 0 )
        memcpy( smWeights, rdpWeights, sizeof(rdpWeights[0])*nWeights );

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
    T_PREC * const smTargetRow = &smBuffer[ bufferSize - 2*N*nColsCacheLine ];
    if ( threadIdx.y == 0 and not iAmSleeping )
    {
        const T_PREC upperBorderValue = data[ threadIdx.x ];
        for ( unsigned iB = 0; iB < N*nColsCacheLine; iB += nColsCacheLine )
            smTargetRow[ iB+threadIdx.x ] = upperBorderValue;
    }


    /* Loop over and calculate the rows. If rnDataY == blockDim.y, then the
     * buffer will exactly suffice, meaning the loop will only be run 1 time */
    for ( T_PREC * curDataRow = data; curDataRow < &data[rnDataX*rnDataY];
          curDataRow += blockDim.y * rnDataX )
    {
        /* move last N rows to the front of the buffer */
        __syncthreads();
        /* @todo: memcpy doesnt respect iAmSleeping yet!
        assert( smTargetRow + N*nColsCacheLine < smBuffer[ bufferSize ] );
        if ( threadIdx.y == 0 and threadIdx.x == 0 )
            memcpy( smBuffer, smTargetRow, N*nColsCacheLine*sizeof(smBuffer[0]) );
        */
        /* memcpy version above parallelized. @todo: benchmark what is faster! */
        if ( threadIdx.y == 0 and not iAmSleeping )
        {
            for ( unsigned iB = 0; iB < N*nColsCacheLine; iB += nColsCacheLine )
                smBuffer[ iB+threadIdx.x ] = smTargetRow[ iB+threadIdx.x ];
        }

        /* Load blockDim.y + N rows into buffer.
         * If data end reached, fill buffer rows with last row
         *   a) Load blockDim.y rows in parallel */
        T_PREC * const pLastData = &data[ (rnDataY-1)*rnDataX + threadIdx.x ];
        const unsigned iBuf = /*skip first N rows*/ N * nColsCacheLine
                            + threadIdx.y * nColsCacheLine + threadIdx.x;
        __syncthreads();
        if ( not iAmSleeping )
        {
            T_PREC * const datum = ptrMin(
                &curDataRow[ threadIdx.y * rnDataX + threadIdx.x ],
                pLastData
            );
            smBuffer[iBuf] = *datum;
        }
        /*   b) Load N rows by master threads, because nThreads >= N is not
         *      guaranteed. */
        if ( not iAmSleeping and threadIdx.y == 0 )
        {
            for ( unsigned iBufRow = N+blockDim.y; iBufRow < nRowsCacheLine; ++iBufRow )
            {
                T_PREC * const datum = ptrMin(
                    &curDataRow[ (iBufRow-N) * rnDataX + threadIdx.x ],
                    pLastData
                );
                smBuffer[ iBufRow*nColsCacheLine + threadIdx.x ] = *datum;
            }
        }
        __syncthreads();

        /* calculated weighted sum on inner rows in buffer, but only if
         * the value we are at is inside the image */
        if ( ( not iAmSleeping )
             and &curDataRow[ threadIdx.y*rnDataX ] < &rdpData[ rnDataX*rnDataY ] )
        {
            T_PREC sum = T_PREC(0);
            /* this for loop is done by each thread and should for large
             * enough kernel sizes sufficiently utilize raw computing power */
            T_PREC * w = smWeights;
            T_PREC * x = &smBuffer[ threadIdx.y * nColsCacheLine + threadIdx.x ];
            for ( ; w < &smWeights[nWeights]; ++w, x += nColsCacheLine )
                sum += (*w) * (*x);
            /* write result back into memory (in-place). No need to wait for
             * all threads to finish, because we write into global memory, to
             * values we already buffered into shared memory! */
            curDataRow[ threadIdx.y * rnDataX + threadIdx.x ] = sum;
        }
    }

}



template<class T_PREC>
void cudaGaussianBlurVertical
( T_PREC * rdpData, int rnDataX, int rnDataY, double rSigma )
{

    /* calculate Gaussian kernel */
    const int nKernelElements = 64;
    T_PREC pKernel[64];
    const int kernelSize = calcGaussianKernel( rSigma, pKernel, nKernelElements );
    assert( kernelSize <= nKernelElements );
    assert( kernelSize % 2 == 1 );

    /* upload kernel to GPU */
    T_PREC * dpKernel;
    CUDA_ERROR( cudaMalloc( &dpKernel, sizeof(T_PREC)*kernelSize ) );
    CUDA_ERROR( cudaMemcpy(  dpKernel, pKernel, sizeof(T_PREC)*kernelSize, cudaMemcpyHostToDevice ) );

    /**
     * the image must be at least nThreadsX threads wide, else many threads
     * will only sleep. The number of blocks is ceil( image height / nThreadsX )
     * Every block works on nThreadsX image columns.
     * Those columns use nThreadsY threads to parallelize the calculation per
     * column.
     * The number of Threads is limited by the hardware to be e.g. 512 or 1024.
     * The reason for this is the limited shared memory size!
     * nThreadsX should be a multiple of a cache line / superword = 32 warps *
     * 1 float per warp = 128 Byte => nThreadsX = 32. For double 16 would also
     * suffice.
     */
    dim3 nThreads( 32, 256/32, 1 );
    dim3 nBlocks ( 1, 1, 1 );
    nBlocks.x = (unsigned) ceilf( (float) rnDataX / nThreads.x );
    const unsigned kernelHalfSize = (kernelSize-1)/2;
    const unsigned bufferSize     = nThreads.x*( nThreads.y + 2*kernelHalfSize );

    cudaKernelApplyKernelVertically<<<
        nBlocks,nThreads,
        sizeof( dpKernel[0] ) * ( kernelSize + bufferSize )
    >>>( rdpData, rnDataX, rnDataY, dpKernel, kernelHalfSize );
    CUDA_ERROR( cudaDeviceSynchronize() );

    CUDA_ERROR( cudaFree( dpKernel ) );
}



template<class T_PREC>
void cudaGaussianBlur
( T_PREC * rData, int rnDataX, int rnDataY, double rSigma )
{
    cudaGaussianBlurHorizontal( rData,rnDataX,rnDataY,rSigma );
    cudaGaussianBlurVertical  ( rData,rnDataX,rnDataY,rSigma );
}



/* Explicitely instantiate certain template arguments to make an object file */
template void cudaApplyKernel<float>( float * const rData, const unsigned rnData, const float * const rWeights, const unsigned rnWeights, const unsigned rnThreads );
template void cudaGaussianBlur<float>( float * rData, int rnData, double rSigma );
template void cudaGaussianBlur<float>( float * rData, int rnDataX, int rnDataY, double rSigma );
template void cudaGaussianBlurHorizontal<float >( float  * rData, int rnDataX, int rnDataY, double rSigma );
template void cudaGaussianBlurVertical<float >( float  * rData, int rnDataX, int rnDataY, double rSigma );

/* @todo: multiple instantations doesn't work, because of extern __shared__ !!!!!! */
/*
template void cudaApplyKernel<double>( double * const rData, const unsigned rnData, const double * const rWeights, const unsigned rnWeights, const unsigned rnThreads );
template void cudaGaussianBlur<double>( double * rData, int rnData, double rSigma );
template void cudaGaussianBlur<double>( double * rData, int rnDataX, int rnDataY, double rSigma );
template void cudaGaussianBlurHorizontal<double>( double * rData, int rnDataX, int rnDataY, double rSigma );
template void cudaGaussianBlurVertical<double>( double * rData, int rnDataX, int rnDataY, double rSigma );
*/

} // namespace image
} // namespace math
} // namespace imresh
