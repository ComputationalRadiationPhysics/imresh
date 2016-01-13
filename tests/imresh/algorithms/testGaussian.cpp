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


#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstdlib>   // srand, rand
#include <cstdint>   // uint32_t, uint64_t
#include <cstring>   // memcpy
#include <chrono>
#include <vector>
#include <cmath>
#include <cfloat>    // FLT_MAX, FLT_EPSILON
#include <cuda_runtime.h>
#include "algorithms/vectorReduce.hpp"
#include "algorithms/cuda/cudaGaussian.h"
#include "libs/cudacommon.h"
#include "benchmarkHelper.hpp"


namespace imresh
{
namespace algorithms
{


    const unsigned nMaxElements = 1024*1024;
    constexpr int maxKernelWidth = 30; // (sigma=4), needed to calculate upper bound of maximum rounding error


    void compareFloatArray
    ( float * pData, float * pResult, int nCols, int nRows, float sigma, bool free = false )
    {
        const int nElements = nCols * nRows;
        auto maxError = vectorMaxAbsDiff( pData, pResult, nElements );
        auto maxValue = vectorMaxAbs( pData, nElements );
        const bool errorMarginOk = maxError / maxValue <= FLT_EPSILON * maxKernelWidth;
        if ( not errorMarginOk )
        {
            std::cout << "Max Error for " << nCols << " columns " << nRows
                      << " rows at sigma=" << sigma << ": " << maxError / maxValue << "\n";

            for ( int iRow = 0; iRow < nRows; ++iRow )
            {
                for ( int iCol = 0; iCol < nCols; ++iCol )
                    std::cout << pData[iRow*nCols+iCol] << " ";
                std::cout << "\n";
            }
            std::cout << " = ? = \n";
            for ( int iRow = 0; iRow < nRows; ++iRow )
            {
                for ( int iCol = 0; iCol < nCols; ++iCol )
                    std::cout << pResult[iRow*nCols+iCol] << " ";
                std::cout << "\n";
            }
            std::cout << std::flush;
        }
        assert( errorMarginOk );
    }

    void checkGaussianHorizontal
    ( float * dpData, float * pData, int nCols, int nRows )
    {
    }
    void checkGaussianVertical
    ( float * dpData, float * pData, int nCols, int nRows )
    {
    }

    void checkIfElementsEqual( float * pData, int nData, int nStride = 1 )
    {
        /* test the maximum divergence of the result vectors, i.e.
         * are they all the same per row ? */
        float sumDiff = 0;
        #pragma omp parallel for reduction( + : sumDiff )
        for ( unsigned i = 1; i < nData; ++i )
            sumDiff += std::abs( pData[ (i-1)*nStride ] - pData[ i*nStride ] );
        //std::cout << "First result element="<<pResult[0]<<", total result divergence=" << sumDiff << "\n";
        assert( sumDiff == 0 );
    }

    void fillWithRandomValues
    ( float * dpData, float * pData, int nElements )
    {
        for ( unsigned i = 0; i < nElements; ++i )
            pData[i] = (float) rand() / RAND_MAX;
        if ( dpData != NULL )
            CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );
    }


    void testGaussian( void )
    {
        using namespace imresh::algorithms::cuda;
        using namespace imresh::libs;

        /* fill test data with random numbers from [-0.5,0.5] */
        float * pData, * dpData, * pResult, * pResultCpu;
        CUDA_ERROR( cudaMallocHost( (void**) &pData, nMaxElements*sizeof(pData[0]) ) );
        CUDA_ERROR( cudaMallocHost( (void**) &pResult, nMaxElements*sizeof(pData[0]) ) );
        CUDA_ERROR( cudaMalloc( (void**)&dpData, nMaxElements*sizeof(dpData[0]) ) );
        pResultCpu = new float[nMaxElements];
        srand(350471643);
        fillWithRandomValues( dpData, pData, nMaxElements );

        /* Test for array of length 1. In this case the values shouldn't change
         * more than from floating point rounding errors */
        std::cout << "Test gaussian blur of single element" << std::flush;
        for ( auto nRows : std::vector<int>{ 1,2,3,5,10,31,37,234,512,1021,1024 } )
        for ( auto sigma : std::vector<float>{ 0.1,0.5,1,1.3,1.7,2,3,4 } )
        {
            //std::cout << "nRows=" << nRows << ", sigma=" << sigma << "\n";
            if ( sigma == 1 )
                std::cout << "." << std::flush;

            /* execute Gaussian blur and test if values unchanged */
            //std::cout << "cudaGaussianBlurHorizontal\n" << std::flush;
            fillWithRandomValues( dpData, pData, nRows );
            cudaGaussianBlurHorizontal( dpData, 1, nRows, sigma );
            CUDA_ERROR( cudaMemcpy( pResult, dpData, nRows*sizeof(dpData[0]), cudaMemcpyDeviceToHost ) );
            compareFloatArray( pData, pResult, 1, nRows, sigma );

            /*** repeat the same checks for vertical blur ***/

            /* execute Gaussian blur and test if values unchanged */
            //std::cout << "cudaGaussianBlurVertical\n" << std::flush;
            fillWithRandomValues( dpData, pData, nRows );
            cudaGaussianBlurVertical( dpData, nRows/*nCols*/, 1, sigma );
            CUDA_ERROR( cudaMemcpy( pResult, dpData, nRows*sizeof(dpData[0]), cudaMemcpyDeviceToHost ) );
            compareFloatArray( pData, pResult, nRows, 1, sigma );

            /*** repeat the same checks for CPU ***/

            /* execute Gaussian blur and test if values unchanged */
            //std::cout << "gaussianBlurHorizontal\n" << std::flush;
            memcpy( pResultCpu, pData, nRows*sizeof(pData[0]) );
            gaussianBlurHorizontal( pResultCpu, 1, nRows, sigma );
            compareFloatArray( pData, pResult, 1, nRows, sigma );
            compareFloatArray( pResultCpu, pResult, 1, nRows, sigma );

            /* execute Gaussian blur and test if values unchanged */
            /* CPU kernel doesn't work for too small image sizes yet! TODO */
            if ( calcGaussianKernel( sigma, (float*)NULL, 0 )/2 < 1 /* rnDataY */ )
            {
                //std::cout << "gaussianBlurVertical\n" << std::flush;
                memcpy( pResultCpu, pData, nRows*sizeof(pData[0]) );
                gaussianBlurVertical( pResultCpu, nRows/*nCols*/, 1, sigma );
                compareFloatArray( pData, pResult, nRows, 1, sigma );
                compareFloatArray( pResultCpu, pResult, nRows, 1, sigma );
            }
        }
        std::cout << "OK\n";

        /* now do checks with longer arrays which contain constant values,
         * meaning the blur shouldn't change them. In fact even accounting for
         * rounding errors every value of the new array should be the same,
         * although they can differ marginally from the original array value */
        std::cout << "Test gaussian blur of vectors of whose elements are all equal" << std::flush;
        float constantValue = 7.37519;
        for ( auto nCols : std::vector<int>{ 1,2,3,5,10,31,37,234,511,512,513,1024,1025 } )
        for ( auto nRows : std::vector<int>{ 1,2,3,5,10,31,37,234,511,512,513,1024,1025 } )
        for ( auto sigma : std::vector<float>{ 0.1,0.5,1,1.7,2,3,4 } )
        {
            const int nElements = nRows*nCols;
            /* skip values accidentally added which are larger then allocated
             * data array length */
            if( nElements > nMaxElements )
                continue;
            float sumDiff = 0;
            if ( nRows == 1 and sigma == 1 )
                std::cout << "." << std::flush;
            //std::cout << "("<<nCols<<","<<nRows<<"), sigma="<<sigma<<"\n";

            /* refresh test data to constant value per row */
            #pragma omp parallel for
            for ( int iRow = 0; iRow < nRows; ++iRow )
                for ( int iCol = 0; iCol < nCols; ++iCol )
                    pData[iRow*nCols+iCol] = (float) iRow/nRows + constantValue;
            CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

            /* execute Gaussian blur and test if values unchanged */
            cudaGaussianBlurHorizontal( dpData, nCols, nRows, sigma );
            CUDA_ERROR( cudaMemcpy( pResult, dpData, nElements*sizeof(dpData[0]), cudaMemcpyDeviceToHost ) );
            compareFloatArray( pResult, pData, nCols, nRows, sigma );
            for ( int iRow = 0; iRow < nRows; ++iRow )
                checkIfElementsEqual( pData+iRow*nCols, nCols );

            /*** repeat the same checks for vertical blur ***/

            /* refresh test data to constant value */
            #pragma omp parallel for
            for ( int iRow = 0; iRow < nRows; ++iRow )
                for ( int iCol = 0; iCol < nCols; ++iCol )
                    pData[iRow*nCols+iCol] = (float) iCol/nCols + constantValue;
            CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

            /* execute Gaussian blur and test if values unchanged */
            cudaGaussianBlurVertical( dpData, nCols, nRows, sigma );
            CUDA_ERROR( cudaMemcpy( pResult, dpData, nElements*sizeof(dpData[0]), cudaMemcpyDeviceToHost ) );
            compareFloatArray( pResult, pData, nCols, nRows, sigma );
            for ( int iCol = 0; iCol < nCols; ++iCol )
                checkIfElementsEqual( pData+iCol, nRows, nCols /*stride*/ );
        }
        std::cout << "OK\n";

        /* test for only 1 element not being 0, then the elements surrounding
         * it should be equal to the kernel (mirrored if asymmetrical) */
         // @TODO

        /* Now test with random data and assert only some general properties:
         *  - mean should be unchanged
         *  - new values shouldn't get larger than largest value of original
         *    data
         *  - the sum of absolute differences to neighboring elements should
         *  - be smaller or equal than in original data (to proof ..)
         *    -> this also should be true at every point!:
         *    data: {x_i} -> gaussian -> {y_i} with y_i = \sum a_k x_{i+k},
         *    0 \leq a_k \leq 1, \sum a_k = 1, k=-m,...,+m, a_{|k|} \leq a_0
         *    now look at |y_{i+1} - y_i| = |\sum a_k [ x_{i+1+k} - x_{i+k} ]|
         *    \leq \sum a_k | x_{i+1+k} - x_{i+k} |
         *    this means element wise this isn't true, because the evening
         *    of (x_i) is equivalent to an evening of (x_i-x_{i-1}) and this
         *    means e.g. for 1,...,1,7,1,0,..,0 the absolute differences are:
         *    0,...,0,6,6,1,0,...,0 so 13 in total
         *    after evening e.g. with kernel 1,1,0 we get:
         *    1,...,1,4,4,0.5,0,...,0 -> diff: 0,...,0,3,0,2.5,0.5,0,...,0
         *    so in 6 in total < 13 (above), but e.g. the difference to the
         *    right changed from 0 to 0.5 or from1 to 2.5, meaning it
         *    increased, because it was next to a very large difference.
         *  - proof for the sum: (assuming periodic values, or compact, i.e.
         *    finite support)
         *    totDiff' = \sum_i |y_{i+1} - y_i|
         *          \leq \sum_i \sum_k a_k | x_{i+1+k} - x_{i+k} |
         *             = \sum_k a_k \sum_i | x_{i+1+k} - x_{i+k} |
         *             = \sum_i | x_{i+1} - x_{i} |
         *    the last step is true because of \sum_k a_k = 1 and because
         *    \sum_i goes over all (periodic) values so that we can group
         *    2*m+1 identical values for \sum_k a_k const to be 1.
         *    (I think this could be shown more formally with some kind of
         *    index tricks)
         *  - we may have used periodic conditions for the proof above, but
         *    the extension of the border introduces only 0 differences and
         *    after the extended values can also be seen as periodic formally,
         *    so that the proof also works for this case.
         */
        const unsigned nRepetitions = 20;
        /* in order to filter out page time outs or similarly long random wait
         * times, we repeat the measurement nRepetitions times and choose the
         * shortest duration measured */

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        using clock = std::chrono::high_resolution_clock;
        float milliseconds, minTime;
        decltype( clock::now() ) clock0, clock1;

        std::cout << "\n";
        std::cout << "Timings in milliseconds:\n";
        std::cout << "image size (nCols,nRows) : cudaGaussianBlurHorizontal | gaussianBlurHorizontal | cudaGaussianBlurVertical | gaussianBlurVertical \n";
        using namespace imresh::tests;
        //for ( auto sigma : std::vector<float>{ 1.5,2,3 } )
        for ( auto sigma : std::vector<float>{ 3 } )
        for ( auto nElements : getLogSpacedSamplingPoints( 2, nMaxElements, 20 ) )
        {
            const int nCols = floor(sqrt( nElements ));
            const int nRows = nCols;
            assert( nCols*nRows <= nElements );
            nElements = nCols*nRows;

            std::cout << std::setw(8) << "("<<nCols<<","<<nRows<<") : ";

            /* time CUDA horizontal blur */
            minTime = FLT_MAX;
            for ( unsigned iRepetition = 0; iRepetition < nRepetitions; ++iRepetition )
            {
                srand(350471643);
                fillWithRandomValues( dpData, pData, nElements );

                cudaEventRecord( start );
                cudaGaussianBlurHorizontal( dpData, nCols, nRows, sigma );
                cudaEventRecord( stop );
                cudaEventSynchronize( stop );

                cudaEventElapsedTime( &milliseconds, start, stop );
                minTime = fmin( minTime, milliseconds );

                CUDA_ERROR( cudaMemcpy( pResult, dpData, nElements*sizeof(dpData[0]), cudaMemcpyDeviceToHost ) );
                checkGaussianHorizontal( pResult, pData, nCols, nRows );
            }
            std::cout << std::setw(8) << minTime << " |" << std::flush;

            /* time CPU horizontal blur */
            minTime = FLT_MAX;
            for ( unsigned iRepetition = 0; iRepetition < nRepetitions; ++iRepetition )
            {
                memcpy( pResultCpu, pData, nElements*sizeof(pData[0]) );
                clock0 = clock::now();
                gaussianBlurHorizontal( pResultCpu, nCols, nRows, sigma );
                clock1 = clock::now();

                milliseconds = ( clock1-clock0 ).count() / 1000.0;
                minTime = fmin( minTime, milliseconds );

                checkGaussianHorizontal( dpData, pData, nCols, nRows );
                compareFloatArray( pResultCpu, pResult, nCols, nRows, sigma );
            }
            std::cout << std::setw(8) << minTime << " |" << std::flush;

            /* time CUDA vertical blur */
            minTime = FLT_MAX;
            for ( unsigned iRepetition = 0; iRepetition < nRepetitions; ++iRepetition )
            {
                srand(350471643);
                fillWithRandomValues( dpData, pData, nElements );

                cudaEventRecord( start );
                cudaGaussianBlurVertical( dpData, nCols, nRows, sigma );
                cudaEventRecord( stop );
                cudaEventSynchronize( stop );

                cudaEventElapsedTime( &milliseconds, start, stop );
                minTime = fmin( minTime, milliseconds );

                CUDA_ERROR( cudaMemcpy( pResult, dpData, nElements*sizeof(dpData[0]), cudaMemcpyDeviceToHost ) );
                checkGaussianVertical( pResult, pData, nCols, nRows );
            }
            std::cout << std::setw(8) << minTime << " |" << std::flush;

            /* time CPU vertical blur */
            minTime = FLT_MAX;
            if ( calcGaussianKernel( sigma, (float*)NULL, 0 )/2 < nRows )
            for ( unsigned iRepetition = 0; iRepetition < nRepetitions; ++iRepetition )
            {
                memcpy( pResultCpu, pData, nElements*sizeof(pData[0]) );
                clock0 = clock::now();
                gaussianBlurVertical( pResultCpu, nCols, nRows, sigma );
                clock1 = clock::now();

                milliseconds = ( clock1-clock0 ).count() / 1000.0;
                minTime = fmin( minTime, milliseconds );

                checkGaussianVertical( dpData, pData, nCols, nRows );
                compareFloatArray( pResultCpu, pResult, nCols, nRows, sigma );
            }
            std::cout << std::setw(8) << minTime << "\n" << std::flush;


        }

        delete[] pResultCpu;
        CUDA_ERROR( cudaFree( dpData ) );
        CUDA_ERROR( cudaFreeHost( pData ) );
        CUDA_ERROR( cudaFreeHost( pResult ) );
    }


} // namespace algorithms
} // namespace imresh


int main( void )
{
    imresh::algorithms::testGaussian();
}
