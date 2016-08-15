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


#include "testGaussian.hpp"

#include <algorithm>    // std::min
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstdlib>      // srand, rand
#include <cstdint>      // uint32_t, uint64_t
#include <cstring>      // memcpy
#include <chrono>
#include <vector>
#include <cmath>
#include <cfloat>       // FLT_MAX, FLT_EPSILON
#include <cuda_runtime.h>
#include <cufft.h>
#ifdef USE_FFTW
#   include <fftw3.h>
#endif

#include "algorithms/vectorReduce.hpp"
#include "algorithms/cuda/cudaGaussian.hpp"
#include "benchmark/imresh/algorithms/cuda/cudaGaussian.hpp"
#include "libs/gaussian.hpp"
#include "libs/calcGaussianKernel.hpp"
#include "libs/cudacommon.hpp"
#include "benchmarkHelper.hpp"      // getLogSpacedSamplingPoints, mean, stddev
#ifdef USE_PNG
#   include "io/readInFuncs/readInFuncs.hpp"
#   include "io/writeOutFuncs/writeOutFuncs.hpp"
#   include <pngwriter.h>
#   include <complex>
#endif


namespace imresh
{
namespace algorithms
{

    void TestGaussian::compareFloatArray
    (
        float const * const pData  ,
        float const * const pResult,
        unsigned int  const nCols  ,
        unsigned int  const nRows  ,
        float         const sigma  ,
        unsigned int  const line   ,
        float         const margin ,
        bool          const print
    )
    {
        const unsigned nElements = nCols * nRows;
        auto maxError = vectorMaxAbsDiff( pData, pResult, nElements );
        float maxValue = vectorMaxAbs( pData, nElements );
        maxValue = fmax( maxValue, vectorMaxAbs( pResult, nElements ) );
        if ( maxValue == 0 )
            maxValue = 1;
        const bool errorMarginOk = margin > 0
                                   ? maxError / maxValue <= margin
                                   : maxError / maxValue <= FLT_EPSILON * maxKernelWidth;
        if ( not errorMarginOk && print )
        {
            std::cout << "Max Error for " << nCols << " columns " << nRows
                      << " rows at sigma=" << sigma << ": " << maxError / maxValue << "\n";

            for ( unsigned iRow = 0; iRow < nRows; ++iRow )
            {
                for ( unsigned iCol = 0; iCol < nCols; ++iCol )
                    std::cout << pData[iRow*nCols+iCol] << " ";
                std::cout << "\n";
            }
            std::cout << " = ? = \n";
            for ( unsigned iRow = 0; iRow < nRows; ++iRow )
            {
                for ( unsigned iCol = 0; iCol < nCols; ++iCol )
                    std::cout << pResult[iRow*nCols+iCol] << " ";
                std::cout << "\n";
            }
            std::cout << "Called from line " << line << "\n";
            std::cout << std::flush;
        }
        assert( errorMarginOk );
    }

    /**
     * Makes some general checks which should hold true for gaussian blurs
     *
     *  - mean should be unchanged (only for periodic boundary conditions)
     *  - new values shouldn't get larger than largest value of original
     *    data, smallest shouldn't get smaller.
     *  - the sum of absolute differences to neighboring elements should
     *    be smaller or equal than in original data (to proof ..)
     *    -> this also should be true at every point!:
     *       data: {x_i} -> gaussian -> {y_i} with y_i = \sum a_k x_{i+k},
     *       0 \leq a_k \leq 1, \sum a_k = 1, k=-m,...,+m, a_{|k|} \leq a_0
     *       now look at |y_{i+1} - y_i| = |\sum a_k [ x_{i+1+k} - x_{i+k} ]|
     *       \leq \sum a_k | x_{i+1+k} - x_{i+k} |
     *       this means element wise this isn't true, because the evening
     *       of (x_i) is equivalent to an evening of (x_i-x_{i-1}) and this
     *       means e.g. for 1,...,1,7,1,0,..,0 the absolute differences are:
     *       0,...,0,6,6,1,0,...,0 so 13 in total
     *       after evening e.g. with kernel 1,1,0 we get:
     *       1,...,1,4,4,0.5,0,...,0 -> diff: 0,...,0,3,0,2.5,0.5,0,...,0
     *       so in 6 in total < 13 (above), but e.g. the difference to the
     *       right changed from 0 to 0.5 or from1 to 2.5, meaning it
     *       increased, because it was next to a very large difference.
     *    -> proof for the sum: (assuming periodic values, or compact, i.e.
     *       finite support)
     *       totDiff' = \sum_i |y_{i+1} - y_i|
     *             \leq \sum_i \sum_k a_k | x_{i+1+k} - x_{i+k} |
     *                = \sum_k a_k \sum_i | x_{i+1+k} - x_{i+k} |
     *                = \sum_i | x_{i+1} - x_{i} |
     *       the last step is true because of \sum_k a_k = 1 and because
     *       \sum_i goes over all (periodic) values so that we can group
     *       2*m+1 identical values for \sum_k a_k const to be 1.
     *       (I think this could be shown more formally with some kind of
     *       index tricks)
     *       - we may have used periodic conditions for the proof above, but
     *         the extension of the border introduces only 0 differences and
     *         after the extended values can also be seen as periodic formally,
     *         so that the proof also works for this case.
     *
     * @param[in] pResult blurred data
     * @param[in] pOriginal raw unblurred data
     * @param[in] nElements number of elements to check in pResult and
     *            pOriginal (Both must be nStride*nElements elements large, or
     *            else invalid memory accesses will occur)
     * @param[in] nStride if 1, then contiguous elements will be checked.
     *            Can be useful to check columns in 2D data, by setting
     *            nStride = nCols. Must not be 0
     **/
    void TestGaussian::checkGaussian
    (
        float const * const pResult  ,
        float const * const pOriginal,
        unsigned int  const nElements,
        unsigned int  const nStride
    )
    {
        assert( vectorMin( pOriginal, nElements, nStride )
             <= vectorMin( pResult  , nElements, nStride ) );
        assert( vectorMax( pOriginal, nElements, nStride )
             >= vectorMax( pResult  , nElements, nStride ) );
        assert( vectorMaxAbsDiff( pResult  , pResult  +nStride, nElements-1, nStride )
             <= vectorMaxAbsDiff( pOriginal, pOriginal+nStride, nElements-1, nStride ) );
    }

    /**
     * Calls checkGaussian for every row
     **/
    void TestGaussian::checkGaussianHorizontal
    (
        float const * const pResult  ,
        float const * const pOriginal,
        unsigned int  const nCols    ,
        unsigned int  const nRows
    )
    {
        for ( auto iRow = 0u; iRow < nRows; ++iRow )
            checkGaussian( pResult, pOriginal, nCols );
    }

    /**
     * Calls checkGaussian for every column
     **/
    void TestGaussian::checkGaussianVertical
    (
        float const * const pResult  ,
        float const * const pOriginal,
        unsigned int  const nCols    ,
        unsigned int  const nRows
    )
    {
        for ( unsigned iCol = 0; iCol < nCols; ++iCol )
            checkGaussian( pResult, pOriginal, nRows, nCols );
    }

    void TestGaussian::checkIfElementsEqual
    (
        float const * const pData,
        unsigned int  const nData,
        unsigned int  const nStride
    )
    {
        assert( nStride > 0 );
        /* test the maximum divergence of the result vectors, i.e.
         * are they all the same per row ? */
        float sumDiff = 0;
        #pragma omp parallel for reduction( + : sumDiff )
        for ( unsigned i = 1; i < nData; ++i )
            sumDiff += std::abs( pData[ (i-1)*nStride ] - pData[ i*nStride ] );

        if ( sumDiff != 0 )
        {
            std::cout << "First result element=" << pData[0] << ", "
                      << "total result divergence=" << sumDiff << "\n"
                      << "elements = ";
            for ( unsigned i = 0; i < nData; ++i )
                std::cout << pData[i] << " ";
            std::cout << "\n" << std::flush;
        }
        assert( sumDiff == 0 );
    }

    void TestGaussian::fillWithRandomValues
    (
        float *      const dpData,
        float *      const pData ,
        unsigned int const nElements
    )
    {
        for ( unsigned i = 0; i < nElements; ++i )
            pData[i] = (float) rand() / RAND_MAX - 0.5;
        if ( dpData != NULL )
            CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );
    }


    void TestGaussian::testGaussianDiracDeltas( void )
    {
        using namespace std::chrono;
        using namespace imresh::algorithms::cuda;
        using namespace benchmark::imresh::algorithms::cuda;
        using namespace imresh::libs;

        /* test gaussian blur for 000100010001000 where the number of 0s
         * corresponds to the kernelwidth, so that we should get:
         * 0,0,w_{+1},w_0,w_{-1},0,w_{+1},w_0,w_{-1},0,w_{+1},w_0,w_{-1},0,0 */
        std::cout << "Test gaussian blur of dirac deltas" << std::flush;
        for ( auto sigma : std::vector<float>{ 0.1,0.5,1,1.3,1.7,2,3,4 } )
        {
            constexpr int nKernelElements = 64;
            float pKernel[nKernelElements];
            const unsigned kernelSize = calcGaussianKernel( sigma, (float*) pKernel, nKernelElements );
            const unsigned kernelHalf = (kernelSize-1)/2;

            std::cout << "." << std::flush;

            /*
            std::cout << "kernelSize = " << kernelSize << ", ";
            std::cout << "weights = ";
            for ( int iW = 0; iW < kernelSize; ++iW )
                std::cout << pKernel[iW] << " ";
            std::cout << "\n";
            */

            for ( auto nCols : std::vector<unsigned>{ 1,2,3,5,10,31,37,234,512,1021,1024 } )
            for ( auto nRows : std::vector<unsigned>{ 1,2,3,5,10,31,37,234,512,1021,1024 } )
            {
                const unsigned nElements = nRows*nCols;
                if ( nElements > nMaxElements )
                {
                    std::cout << "Skipping Image Size " << nRows << " x " << nCols << ", because not enough memory allocated, check test image sizes and nMaxElements in source code!\n";
                    continue;
                }

                if ( nCols >= kernelSize )
                {
                    /* initialize data */
                    memset( pData, 0, nElements*sizeof( pData[0] ) );
                    for ( unsigned iRow = 0; iRow < nRows; ++iRow )
                    for ( unsigned iCol = kernelHalf; iCol < nCols - kernelHalf; iCol += kernelSize )
                        pData[ iRow*nCols + iCol ] = 1;

                    /* write down expected solution */
                    for ( unsigned iRow = 0; iRow < nRows; ++iRow )
                    {
                        unsigned iCol = 0;
                        for ( ; iCol + kernelSize-1 < nCols; iCol += kernelSize )
                        {
                            for ( unsigned iW = 0; iW < kernelSize; ++iW )
                                pSolution[ iRow*nCols + iCol+iW ] = pKernel[ iW ];
                        }
                        for ( ; iCol < nCols; ++iCol )
                            pSolution[ iRow*nCols + iCol ] = 0;
                    }

                    /* execute Gaussian blur and test if values unchanged */
                    //std::cout << "cudaGaussianBlurHorizontal\n" << std::flush;
                    CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof( dpData[0] ), cudaMemcpyHostToDevice ) );
                    cudaGaussianBlurHorizontal( dpData, nCols, nRows, sigma );
                    CUDA_ERROR( cudaMemcpy( pResult, dpData, nElements*sizeof( dpData[0] ), cudaMemcpyDeviceToHost ) );
                    compareFloatArray( pResult, pSolution, nCols, nRows, sigma, __LINE__ );

                    CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof( dpData[0] ), cudaMemcpyHostToDevice ) );
                    cudaGaussianBlurHorizontalSharedWeights( dpData, nCols, nRows, sigma );
                    CUDA_ERROR( cudaMemcpy( pResult, dpData, nElements*sizeof( dpData[0] ), cudaMemcpyDeviceToHost ) );
                    compareFloatArray( pResult, pSolution, nCols, nRows, sigma, __LINE__ );

                    /* execute Gaussian blur and test if values unchanged */
                    //std::cout << "gaussianBlurHorizontal\n" << std::flush;
                    memcpy( pResultCpu, pData, nElements*sizeof(pData[0]) );
                    gaussianBlurHorizontal( pResultCpu, nCols, nRows, sigma );
                    compareFloatArray( pResultCpu, pSolution, nCols, nRows, sigma, __LINE__ );
                }

                /*** repeat the same checks for vertical blur ***/

                if ( nRows >= kernelSize )
                {
                    /* initialize data */
                    memset( pData, 0, nElements*sizeof( pData[0] ) );
                    for ( unsigned iRow = kernelHalf; iRow < nRows - kernelHalf; iRow += kernelSize )
                    for ( unsigned iCol = 0; iCol < nCols; ++iCol )
                        pData[ iRow*nCols + iCol ] = 1;

                    /* write down expected solution */
                    unsigned iRow = 0;
                    for ( ; iRow + kernelSize-1 < nRows; iRow += kernelSize )
                    {
                        for ( unsigned iCol = 0; iCol < nCols; ++iCol )
                        {
                            for ( unsigned iW = 0; iW < kernelSize; ++iW )
                                pSolution[ (iRow+iW)*nCols + iCol ] = pKernel[ iW ];
                        }
                    }
                    for ( ; iRow < nRows; ++iRow )
                    for ( unsigned iCol = 0; iCol < nCols; ++iCol )
                        pSolution[ iRow*nCols + iCol ] = 0;

                    /* execute Gaussian blur and test if values unchanged */
                    //std::cout << "cudaGaussianBlurVertical\n" << std::flush;
                    CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof( dpData[0] ), cudaMemcpyHostToDevice ) );
                    cudaGaussianBlurVertical( dpData, nCols, nRows, sigma );
                    CUDA_ERROR( cudaMemcpy( pResult, dpData, nElements*sizeof(dpData[0]), cudaMemcpyDeviceToHost ) );
                    compareFloatArray( pResult, pSolution, nCols, nRows, sigma, __LINE__ );

                    /*** repeat the same checks for CPU ***/

                    /* execute Gaussian blur and test if values unchanged */
                    //std::cout << "gaussianBlurVertical\n" << std::flush;
                    memcpy( pResultCpu, pData, nElements*sizeof(pData[0]) );
                    gaussianBlurVertical( pResultCpu, nCols, nRows, sigma );
                    compareFloatArray( pResult, pSolution, nCols, nRows, sigma, __LINE__ );
                }
            }
        }
        std::cout << "OK\n";
    }


    void TestGaussian::testGaussianRandomSingleData( void )
    {
        using namespace imresh::algorithms::cuda;
        using namespace benchmark::imresh::algorithms::cuda;
        using namespace imresh::libs;

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
            compareFloatArray( pData, pResult, 1, nRows, sigma, __LINE__ );

            cudaGaussianBlurHorizontalSharedWeights( dpData, 1, nRows, sigma );
            CUDA_ERROR( cudaMemcpy( pResult, dpData, nRows*sizeof(dpData[0]), cudaMemcpyDeviceToHost ) );
            compareFloatArray( pData, pResult, 1, nRows, sigma, __LINE__ );

            /*** repeat the same checks for vertical blur ***/

            /* execute Gaussian blur and test if values unchanged */
            //std::cout << "cudaGaussianBlurVertical\n" << std::flush;
            fillWithRandomValues( dpData, pData, nRows );
            cudaGaussianBlurVertical( dpData, nRows/*nCols*/, 1, sigma );
            CUDA_ERROR( cudaMemcpy( pResult, dpData, nRows*sizeof(dpData[0]), cudaMemcpyDeviceToHost ) );
            compareFloatArray( pData, pResult, nRows, 1, sigma, __LINE__ );

            /*** repeat the same checks for CPU ***/

            /* execute Gaussian blur and test if values unchanged */
            //std::cout << "gaussianBlurHorizontal\n" << std::flush;
            memcpy( pResultCpu, pData, nRows*sizeof(pData[0]) );
            gaussianBlurHorizontal( pResultCpu, 1, nRows, sigma );
            compareFloatArray( pData, pResult, 1, nRows, sigma );
            compareFloatArray( pResultCpu, pResult, 1, nRows, sigma, __LINE__ );

            /* execute Gaussian blur and test if values unchanged */
            /* CPU kernel doesn't work for too small image sizes yet! TODO */
            if ( calcGaussianKernel( sigma, (float*)NULL, 0 )/2 < 1 /* rnDataY */ )
            {
                //std::cout << "gaussianBlurVertical\n" << std::flush;
                memcpy( pResultCpu, pData, nRows*sizeof(pData[0]) );
                gaussianBlurVerticalUncached( pResultCpu, nRows/*nCols*/, 1, sigma );
                compareFloatArray( pData, pResult, nRows, 1, sigma, __LINE__ );
                compareFloatArray( pResultCpu, pResult, nRows, 1, sigma, __LINE__ );
            }
        }
        std::cout << "OK\n";
    }


    void TestGaussian::testGaussianConstantValuesPerRowLine( void )
    {
        using namespace imresh::algorithms::cuda;
        using namespace benchmark::imresh::algorithms::cuda;
        using namespace imresh::libs;

        /* now do checks with longer arrays which contain constant values,
         * meaning the blur shouldn't change them. In fact even accounting for
         * rounding errors every value of the new array should be the same,
         * although they can differ marginally from the original array value */
        std::cout << "Test gaussian blur of vectors of whose elements are all equal" << std::flush;
        for ( auto nCols : std::vector<unsigned>{ 1,2,3,5,10,31,37,234,511,512,513,1024,1025 } )
        for ( auto nRows : std::vector<unsigned>{ 1,2,3,5,10,31,37,234,511,512,513,1024,1025 } )
        for ( auto sigma : std::vector<float>{ 0.1,0.5,1,1.7,2,3,4 } )
        {
            float constantValue = 7.37519;

            const unsigned nElements = nRows*nCols;
            /* skip values accidentally added which are larger then allocated
             * data array length */
            if( nElements > nMaxElements )
                continue;
            if ( nRows == 1 and sigma == 1 )
                std::cout << "." << std::flush;
            //std::cout << "("<<nCols<<","<<nRows<<"), sigma="<<sigma<<"\n";

            /* refresh test data to constant value per row */
            #pragma omp parallel for
            for ( unsigned iRow = 0; iRow < nRows; ++iRow )
                for ( unsigned iCol = 0; iCol < nCols; ++iCol )
                    pData[iRow*nCols+iCol] = constantValue;

            /* execute Gaussian blur and test if values unchanged */
            //std::cout << "CUDA horizontal\n";
            CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );
            cudaGaussianBlurHorizontal( dpData, nCols, nRows, sigma );
            CUDA_ERROR( cudaMemcpy( pResult, dpData, nElements*sizeof(dpData[0]), cudaMemcpyDeviceToHost ) );
            compareFloatArray( pResult, pData, nCols, nRows, sigma, __LINE__ );
            checkIfElementsEqual( pData, nRows*nCols );

            CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );
            cudaGaussianBlurHorizontalSharedWeights( dpData, nCols, nRows, sigma );
            CUDA_ERROR( cudaMemcpy( pResult, dpData, nElements*sizeof(dpData[0]), cudaMemcpyDeviceToHost ) );
            compareFloatArray( pResult, pData, nCols, nRows, sigma, __LINE__ );
            checkIfElementsEqual( pData, nRows*nCols );

            /* do the same on CPU and compare */
            //std::cout << "CPU horizontal\n";
            memcpy( pResultCpu, pData, nRows*nCols*sizeof( pData[0] ) );
            gaussianBlurHorizontal( pResultCpu, nCols, nRows, sigma );
            checkIfElementsEqual( pResultCpu, nRows*nCols );
            compareFloatArray( pResultCpu, pData, nCols, nRows, sigma, __LINE__ );
            compareFloatArray( pResultCpu, pResult, nCols, nRows, sigma, __LINE__ );

            /*** repeat the same checks for vertical blur ***/

            /* refresh test data to constant value */
            #pragma omp parallel for
            for ( unsigned iRow = 0; iRow < nRows; ++iRow )
                for ( unsigned iCol = 0; iCol < nCols; ++iCol )
                    pData[iRow*nCols+iCol] = constantValue;
            CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

            /* execute Gaussian blur and test if values unchanged */
            //std::cout << "CUDA vertical\n";
            cudaGaussianBlurVertical( dpData, nCols, nRows, sigma );
            CUDA_ERROR( cudaMemcpy( pResult, dpData, nElements*sizeof(dpData[0]), cudaMemcpyDeviceToHost ) );
            compareFloatArray( pResult, pData, nCols, nRows, sigma, __LINE__ );
            checkIfElementsEqual( pData, nRows*nCols );

            /* do the same on CPU and compare */
            //std::cout << "CPU vertical\n";
            memcpy( pResultCpu, pData, nRows*nCols*sizeof( pData[0] ) );
            gaussianBlurVertical( pResultCpu, nCols, nRows, sigma );
            checkIfElementsEqual( pResultCpu, nRows*nCols );
            compareFloatArray( pResultCpu, pData, nCols, nRows, sigma, __LINE__ );
            compareFloatArray( pResultCpu, pResult, nCols, nRows, sigma, __LINE__ );

        }
        std::cout << "OK\n";
    }


    void TestGaussian::testGaussianConstantValues( void )
    {
        using namespace imresh::algorithms::cuda;
        using namespace benchmark::imresh::algorithms::cuda;
        using namespace imresh::libs;

        std::cout << "Test gaussian blur of vectors of whose rows or columns (depending on whether to use vertical or horizontal blur) are all equal" << std::flush;
        for ( auto nCols : std::vector<unsigned>{ 1,2,3,5,10,31,37,234,511,512,513,1024,1025 } )
        for ( auto nRows : std::vector<unsigned>{ 1,2,3,5,10,31,37,234,511,512,513,1024,1025 } )
        for ( auto sigma : std::vector<float>{ 0.1,0.5,1,1.7,2,3,4 } )
        {
            float constantValue = 7.37519;

            const unsigned nElements = nRows*nCols;
            /* skip values accidentally added which are larger then allocated
             * data array length */
            if( nElements > nMaxElements )
                continue;
            if ( nRows == 1 and sigma == 1 )
                std::cout << "." << std::flush;
            //std::cout << "("<<nCols<<","<<nRows<<"), sigma="<<sigma<<"\n";

            /* refresh test data to constant value per row */
            #pragma omp parallel for
            for ( unsigned iRow = 0; iRow < nRows; ++iRow )
                for ( unsigned iCol = 0; iCol < nCols; ++iCol )
                    pData[iRow*nCols+iCol] = (float) iRow/nRows + constantValue;

            /* execute Gaussian blur and test if values unchanged */
            CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );
            cudaGaussianBlurHorizontal( dpData, nCols, nRows, sigma );
            CUDA_ERROR( cudaMemcpy( pResult, dpData, nElements*sizeof(dpData[0]), cudaMemcpyDeviceToHost ) );
            compareFloatArray( pResult, pData, nCols, nRows, sigma, __LINE__ );
            for ( unsigned iRow = 0; iRow < nRows; ++iRow )
                checkIfElementsEqual( pData+iRow*nCols, nCols );

            CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );
            cudaGaussianBlurHorizontalSharedWeights( dpData, nCols, nRows, sigma );
            CUDA_ERROR( cudaMemcpy( pResult, dpData, nElements*sizeof(dpData[0]), cudaMemcpyDeviceToHost ) );
            compareFloatArray( pResult, pData, nCols, nRows, sigma, __LINE__ );
            for ( unsigned iRow = 0; iRow < nRows; ++iRow )
                checkIfElementsEqual( pData+iRow*nCols, nCols );

            /*** repeat the same checks for vertical blur ***/

            /* refresh test data to constant value */
            #pragma omp parallel for
            for ( unsigned iRow = 0; iRow < nRows; ++iRow )
                for ( unsigned iCol = 0; iCol < nCols; ++iCol )
                    pData[iRow*nCols+iCol] = (float) iCol/nCols + constantValue;
            CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

            /* execute Gaussian blur and test if values unchanged */
            cudaGaussianBlurVertical( dpData, nCols, nRows, sigma );
            CUDA_ERROR( cudaMemcpy( pResult, dpData, nElements*sizeof(dpData[0]), cudaMemcpyDeviceToHost ) );
            compareFloatArray( pResult, pData, nCols, nRows, sigma, __LINE__ );
            for ( unsigned iCol = 0; iCol < nCols; ++iCol )
                checkIfElementsEqual( pData+iCol, nRows, nCols /*stride*/ );
        }
        std::cout << "OK\n";

    }


    void TestGaussian::benchmarkGaussianGeneralRandomValues( void )
    {
        using namespace std::chrono;
        using namespace imresh::algorithms::cuda;
        using namespace benchmark::imresh::algorithms::cuda;
        using namespace imresh::libs;

        /* Now test with random data and assert only some general properties */
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

        std::cout
            << "\nGaussian blur timings in milliseconds:\n"
            << "image size (nCols,nRows) "
            << ": CUDA const memory horizontal "
            << "| CUDA shared memory horizontal "
            << "| CPU horizontal "
            << "| CUDA vertical "
            << "| CPU vertical "
            << "| CPU vertical with software cache"
            << "| CUDA FFT"
            << "| CPU FFT"
            << std::endl;
        using namespace imresh::tests;
        //for ( auto sigma : std::vector<float>{ 1.5,2,3 } )
        for ( auto sigma : std::vector<float>{ 3 } )
        for ( auto nElements : getLogSpacedSamplingPoints( 2, nMaxElements, 20 ) )
        {
            const unsigned nCols = floor(sqrt( nElements ));
            const unsigned nRows = nCols;
            assert( nCols*nRows <= nMaxElements );
            nElements = nCols*nRows;

            std::cout << "(" << std::setw(5) << nCols << ","
                             << std::setw(5) << nRows << ") : ";

            #define TIME_GPU( FUNC, CHECK )                                    \
            {                                                                  \
                minTime = FLT_MAX;                                             \
                for ( auto iRepetition = 0u; iRepetition < nRepetitions;       \
                      ++iRepetition )                                          \
                {                                                              \
                    srand(350471643);                                          \
                    fillWithRandomValues( dpData, pData, nElements );          \
                                                                               \
                    cudaEventRecord( start );                                  \
                    FUNC( dpData, nCols, nRows, sigma );                       \
                    cudaEventRecord( stop );                                   \
                    cudaEventSynchronize( stop );                              \
                                                                               \
                    cudaEventElapsedTime( &milliseconds, start, stop );        \
                    minTime = fmin( minTime, milliseconds );                   \
                                                                               \
                    CUDA_ERROR( cudaMemcpy( pResult, dpData, nElements *       \
                                sizeof(dpData[0]), cudaMemcpyDeviceToHost ) ); \
                    CHECK( pResult, pData, nCols, nRows );                     \
                }                                                              \
                std::cout << std::setw(8) << minTime << " |" << std::flush;    \
            }

            #define NOP( ... )

            #define TIME_CPU( FUNC, CHECK )                               \
            minTime = FLT_MAX;                                            \
            for ( unsigned iRepetition = 0; iRepetition < nRepetitions;   \
                  ++iRepetition )                                         \
            {                                                             \
                memcpy( pResultCpu, pData, nElements*sizeof(pData[0]) );  \
                clock0 = clock::now();                                    \
                FUNC( pResultCpu, nCols, nRows, sigma );                  \
                clock1 = clock::now();                                    \
                                                                          \
                auto seconds = duration_cast<duration<double>>(           \
                                    clock1 - clock0 );                    \
                minTime = fmin( minTime, seconds.count() * 1000 );        \
                                                                          \
                CHECK( pResultCpu, pData, nCols, nRows );                 \
                compareFloatArray( pResultCpu, pResult, nCols, nRows,     \
                                   sigma, __LINE__ );                     \
            }                                                             \
            std::cout << std::setw(8) << minTime << " |" << std::flush;

            TIME_GPU( cudaGaussianBlurHorizontalConstantWeights, checkGaussianHorizontal )
            TIME_GPU( cudaGaussianBlurHorizontalSharedWeights  , checkGaussianHorizontal )

            TIME_CPU( gaussianBlurHorizontal  , checkGaussianHorizontal )
            TIME_GPU( cudaGaussianBlurVertical, checkGaussianVertical )

            if ( (unsigned) calcGaussianKernel( sigma, (float*)NULL, 0 )/2 < nRows )
            {
                TIME_CPU( gaussianBlurVerticalUncached, checkGaussianVertical )
            }
            else
            {
                std::cout << std::setw(8) << "-" << " |";
            }

            TIME_CPU( gaussianBlurVertical, checkGaussianVertical )
            TIME_GPU( cudaGaussianBlurVertical, NOP )
            TIME_CPU( gaussianBlurVertical, NOP )

            std::cout << std::endl;

            #undef TIME_GPU
            #undef TIME_CPU
            #undef NOP
        }
    }

    void TestGaussian::testFourierConvolution( void )
    {
        using namespace imresh::io::writeOutFuncs;
        using namespace imresh::libs;   // calcGaussianKernel2d, gaussianBlur

        auto n = 256u;
        HostDeviceMemory<float> image( n*n ), kernel( n*n );
        calcGaussianKernel2d( 10, 50,50, kernel.cpu, n,n );
        writeOutPNG( kernel.cpu, n,n, "extendedKernel.png" );

        /**
         * create a triangle in the center of the image to blur
         * Use 4 points and Wiegner-Seitz cell definition to draw it:
         * @verbatim
         * y
         * ^
         * |             '..'
         * |         o 3 .''.    o 2
         * |           .'    '.
         * |         .'   o 0  '.
         * |       .'            '.
         * |   --.'----------------'.---
         * |   .'                    '.
         * | .'           o 1          '.
         * |
         * +-------------------------------> x
         * @endverbatim
         */
        unsigned int
            x0 = n/2, x1 = n/2, x2 = 0.75 * n, x3 = 0.25*n,
            y0 = n/2, y1 = n/4, y2 = 0.75 * n, y3 = 0.75*n;
        for ( auto ix = 0u; ix < n; ++ix )
        for ( auto iy = 0u; iy < n; ++iy )
        {
            unsigned int
                ds0 = (ix-x0)*(ix-x0) + (iy-y0)*(iy-y0),
                ds1 = (ix-x1)*(ix-x1) + (iy-y1)*(iy-y1),
                ds2 = (ix-x2)*(ix-x2) + (iy-y2)*(iy-y2),
                ds3 = (ix-x3)*(ix-x3) + (iy-y3)*(iy-y3);
            image.cpu[ iy*n + ix ] =
                std::min( { ds0, ds1, ds2, ds3 } ) == ds0 ? 1.f : 0.f;
        }
        plotPng( image.cpu, n,n, "toBlur.png" );

        #ifdef USE_FFTW
        {
            auto pImageFt  = new fftwf_complex[ n*n ];
            auto pKernelFt = new fftwf_complex[ n*n ];
            auto tmp       = new float        [ n*n ];

            {
                for ( auto i = 0u; i < n*n; ++i )
                {
                    pImageFt[i][0] = image.cpu[i];
                    pImageFt[i][1] = 0;
                }
                auto cpuFtPlan = fftwf_plan_dft_2d( n, n, pImageFt, pImageFt,
                                                    FFTW_FORWARD, FFTW_ESTIMATE );
                fftwf_execute( cpuFtPlan );
                plotComplexPng( pImageFt, n,n, "toBlurFt.png", false, false, 2 );
                plotComplexPng( pImageFt, n,n, "toBlurFtSwapped.png", true, true, 2 );

                for ( auto i = 0u; i < n*n; ++i )
                    tmp[i] = pImageFt[i][0];
                writeOutPNG( tmp, n,n, "toBlurFtReal.png" );
                for ( auto i = 0u; i < n*n; ++i )
                    tmp[i] = pImageFt[i][1];
                writeOutPNG( tmp, n,n, "toBlurFtImag.png" );
            }

            /* same with pImageFt -> pKernelFt -> use function ? */
            {
                for ( auto i = 0u; i < n*n; ++i )
                {
                    pKernelFt[i][0] = kernel.cpu[i];
                    pKernelFt[i][1] = 0;
                }
                auto cpuFtPlan = fftwf_plan_dft_2d( n, n, pKernelFt, pKernelFt,
                                                    FFTW_FORWARD, FFTW_ESTIMATE );
                fftwf_execute( cpuFtPlan );
                plotComplexPng( pKernelFt, n,n, "extendedKernelFt.png", false, false, 2 );
                plotComplexPng( pKernelFt, n,n, "extendedKernelFtSwapped.png", true, true, 2 );

                for ( auto i = 0u; i < n*n; ++i )
                    tmp[i] = pKernelFt[i][0];
                writeOutPNG( tmp, n,n, "extendedKernelFtReal.png" );
                for ( auto i = 0u; i < n*n; ++i )
                    tmp[i] = pKernelFt[i][1];
                writeOutPNG( tmp, n,n, "extendedKernelFtImag.png" );
            }

            for ( auto i = 0u; i < n*n; ++i )
            {
                /* (a+ib)*(c+id) = ac-bd + i(ad+bc) */
                float const /* not ref! */
                    a = pImageFt[i][0], c = pKernelFt[i][0],
                    b = pImageFt[i][1], d = pKernelFt[i][1];
                pImageFt[i][0] = a*c - b*d;
                pImageFt[i][1] = a*d + b*c;
            }
            plotComplexPng( pImageFt, n,n, "multiplied.png", true, true, 2 );

            auto cpuFtPlan = fftwf_plan_dft_2d( n, n, pImageFt, pImageFt,
                                                FFTW_BACKWARD, FFTW_ESTIMATE );
            fftwf_execute( cpuFtPlan );
            plotComplexPng( pImageFt, n,n, "blurred.png" );

            for ( auto i = 0u; i < n*n; ++i )
                tmp[i] = pImageFt[i][0];
            plotPng( tmp, n,n, "blurredReal.png" );
            for ( auto i = 0u; i < n*n; ++i )
                tmp[i] = pImageFt[i][1];
            plotPng( tmp, n,n, "blurredImag.png" );

            float maxReal = 0;
            for ( auto i = 0u; i < n*n; ++i )
                maxReal = std::max( maxReal, std::abs( pImageFt[i][0] ) );
            std::cout << "max value of the blurred image is " << maxReal << "\n"
                      << "max value of the original image was 1.0\n";

            delete[] pImageFt;
            delete[] pKernelFt;
            delete[] tmp;
        }
        #endif

        //cufftHandle gpuFtPlan;
        //CUFFT_ERROR( cufftPlan2d( &gpuFtPlan, Ny /* nRows */, Nx /* nColumns */, CUFFT_C2C ) );
        //CUFFT_ERROR( cufftExecC2C( gpuFtPlan, dpData, dpResult, CUFFT_FORWARD ) );
        //writeOutPNG( image.cpu, n,n, "extendedKernelFt.png" );

        /* Test if the converted Gaussian kernel is exactly as the theory
         * predicts, because that could save one convolution when directly
         * calculating the frequency space kernel */
        float sigma = 10;
        calcGaussianKernel2d( sigma, 0,0, kernel.cpu, n,n );
        writeOutPNG( kernel.cpu, n,n, "extendedKernel.png" );
        {
            float maxValue = 0;
            float minValue = INFINITY;
            float sumKernel = 0;
            for ( auto i = 0u; i < n*n; ++i )
            {
                minValue = std::min( minValue, kernel.cpu[i] );
                maxValue = std::max( maxValue, kernel.cpu[i] );
                sumKernel += kernel.cpu[i];
            }
            float calcSigma = 0;
            float const gaussAtSigma = std::exp( - 1./2 ); // 0.60653
            for ( auto i = 0u; i < n-1; ++i ) // only first row
            {
                if ( ( kernel.cpu[i  ] >= maxValue * gaussAtSigma ) &&
                     ( kernel.cpu[i+1] <= maxValue * gaussAtSigma )   )
                {
                    /* interpolate where it actually is equal
                     * c := maxValue * gaussAtSigma:
                     * f(x) = ai + ( aip1 - ai ) / ( (i+1) - i ) * ( x - i )
                     * f(x) != c => x = ( c - ai ) / ( aip1 - ai ) + i
                     */
                    calcSigma = ( maxValue * gaussAtSigma - kernel.cpu[i] ) /
                                ( kernel.cpu[i+1] - kernel.cpu[i] ) + i;
                    break;
                }
            }
            std::cout
                << "Statistics for the original Gaussian Kernel:\n"
                << "    sigma         : " << sigma    << "\n"
                << "    minimum value : " << minValue << "\n"
                << "    maximum value : " << maxValue << "\n"
                << "    1/(2*pi*s*s)  : " << 1.0/( 2.0 * M_PI * sigma * sigma ) << " (should be equal to maximum value)\n"
                << "    total sum     : " << sumKernel << " (should be equal 1.0)\n"
                << "    calc sigma    : " << calcSigma << " (should be sigma)\n"
                //<< "    stddev iy=0   : " << imresh::tests::stddev( kernel.cpu, n ) * n << " (should be sigma)\n"
                << "from the analytically calculated transformed kernel" << std::endl;
        }

        #ifdef USE_FFTW
        {
            auto pKernelFt          = new fftwf_complex[ n*n ];
            auto pKernelFtPredicted = new fftwf_complex[ n*n ];
            for ( auto i = 0u; i < n*n; ++i )
            {
                pKernelFt[i][0] = kernel.cpu[i];
                pKernelFt[i][1] = 0;
            }
            auto cpuFtPlan = fftwf_plan_dft_2d( n, n, pKernelFt, pKernelFt,
                                                FFTW_FORWARD, FFTW_ESTIMATE );
            fftwf_execute( cpuFtPlan );
            plotComplexPng( pKernelFt, n,n, "NonTranslatedKernelFtSwapped.png",
                            true, true, 2 );

            /**
             * Calculate using analytical formula for mu=0:
             * f(kx,ky) = 1/sqrt(2*pi) * exp[ -(kx^2+ky^2)*sigma^2 ]
             */
            calcGaussianKernel2d( 1.*sqrt(n)/sigma, 0,0, kernel.cpu, n,n );
            for ( auto i = 0u; i < n*n; ++i )
            {
                pKernelFtPredicted[i][0] = n * sigma * kernel.cpu[i];
                pKernelFtPredicted[i][1] = 0;
            }
                //pKernelFtPredicted[iy*n+ix][0] = 1./std::sqrt(2.*M_PI) *
                 //                                std::exp( -(iy*iy+ix*ix)/(sigma*sigma) );
            plotComplexPng( pKernelFtPredicted, n,n, "NonTranslatedKernelFtPredicted.png",
                            false, false, 2 );
            plotComplexPng( pKernelFtPredicted, n,n, "NonTranslatedKernelFtPredictedSwapped.png",
                            true, true, 2 );

            /* Calculate deviation */
            float
                maxAbsErrRe = 0, maxAbsRe = 0, maxRelErrRe = 0,
                maxAbsErrIm = 0, maxAbsIm = 0, maxRelErrIm = 0,
                maxAbsReOrig = 0, minAbsReOrig = INFINITY,
                maxAbsRePred = 0, minAbsRePred = INFINITY,
                maxReOrig    = 0, minReOrig    = INFINITY,
                maxRePred    = 0, minRePred    = INFINITY,
                sumOrig = 0, sumPred = 0;
            for ( auto i = 0u; i < n*n; ++i )
            {
                maxAbsReOrig = std::max( maxAbsReOrig, std::abs( pKernelFt         [i][0] ) );
                maxAbsRePred = std::max( maxAbsRePred, std::abs( pKernelFtPredicted[i][0] ) );

                minAbsReOrig = std::min( minAbsReOrig, std::abs( pKernelFt         [i][0] ) );
                minAbsRePred = std::min( minAbsRePred, std::abs( pKernelFtPredicted[i][0] ) );

                maxReOrig = std::max( maxReOrig, pKernelFt         [i][0] );
                maxRePred = std::max( maxRePred, pKernelFtPredicted[i][0] );

                minReOrig = std::min( minReOrig, pKernelFt         [i][0] );
                minRePred = std::min( minRePred, pKernelFtPredicted[i][0] );

                sumOrig += pKernelFt         [i][0];
                sumPred += pKernelFtPredicted[i][0];

                float const maxValRe = std::max(
                    std::abs( pKernelFtPredicted[i][0] ),
                    std::abs( pKernelFt         [i][0] )
                );
                float const maxValIm = std::max(
                    std::abs( pKernelFtPredicted[i][1] ),
                    std::abs( pKernelFt         [i][1] )
                );
                maxAbsRe = std::max( maxAbsRe, maxValRe );
                maxAbsIm = std::max( maxAbsIm, maxValIm );

                pKernelFtPredicted[i][0] -= pKernelFt[i][0];
                pKernelFtPredicted[i][1] -= pKernelFt[i][1];

                maxAbsErrRe = std::max( maxAbsErrRe, std::abs( pKernelFtPredicted[i][0] ) );
                maxAbsErrIm = std::max( maxAbsErrIm, std::abs( pKernelFtPredicted[i][1] ) );
                maxRelErrRe = std::max( maxRelErrRe,
                                        std::abs( pKernelFtPredicted[i][0] ) / maxValRe );
                maxRelErrIm = std::max( maxRelErrIm,
                                        std::abs( pKernelFtPredicted[i][1] ) / maxValIm );
            }
            float calcSigmaPred = 0;
            float const gaussAtSigma = std::exp( - 1./2 ); // 0.60653
            for ( auto i = 0u; i < n-1; ++i ) // only first row
            {
                auto const ai  = pKernelFt[i  ][0];
                auto const ai1 = pKernelFt[i+1][0];
                std::cout << ai << " ";
                if ( ( ai  >= maxAbsRePred * gaussAtSigma ) &&
                     ( ai1 <= maxAbsRePred * gaussAtSigma )   )
                {
                    /* interpolate where it actually is equal
                     * c := maxValue * gaussAtSigma:
                     * f(x) = ai + ( aip1 - ai ) / ( (i+1) - i ) * ( x - i )
                     * f(x) != c => x = ( c - ai ) / ( aip1 - ai ) + i
                     */
                    calcSigmaPred = ( maxAbsRePred * gaussAtSigma - ai ) / ( ai1 - ai ) + i;
                    std::cout << ai1 << " (sought for: " << maxAbsRePred * gaussAtSigma << ")" << std::endl;
                    break;
                }
            }
            plotComplexPng( pKernelFtPredicted, n,n, "NonTranslatedFtPredictedDeviationSwapped.png",
                            true, true, 2 );
            std::cout
                << "The transformed Gaussian Kernel deviates:\n"
                << "    Re: abs. diff.     : " << maxAbsErrRe   << "\n"
                << "        min abs.orig.  : " << minAbsReOrig  << "\n"
                << "        max abs.orig.  : " << maxAbsReOrig  << "\n"
                << "        min abs.pred.  : " << minAbsRePred  << "\n"
                << "        max abs.pred.  : " << maxAbsRePred  << "\n"
                << "        min     orig.  : " << minReOrig     << "\n"
                << "        max     orig.  : " << maxReOrig     << "\n"
                << "        min     pred.  : " << minRePred     << "\n"
                << "        max     pred.  : " << maxRePred     << "\n"
                << "        max rel. error : " << maxRelErrRe   << "\n"
                << "    Im: abs. diff.     : " << maxAbsErrIm   << "\n"
                << "        max abs.       : " << maxAbsIm      << "\n"
                << "        max rel. error : " << maxRelErrIm   << "\n"
                << "    calc sigma         : " << calcSigmaPred << "\n"
                << "from the analytically calculated transformed kernel" << std::endl;
        }
        #endif

        #ifdef USE_FFTW
        /**
         * Compare convolution blur, with direct blur
         */
        std::cout << "Check old blur function with FFT blur ";
        for ( auto n : std::vector<unsigned>{ 1,2,3,5,10,31,37,234,511,512,513,1024,1025 } )
        for ( auto sigma : std::vector<float>{ 0.1,0.5,1,1.7,2,3,4,10 } )
        {
            auto blurRaw = new float[ n*n ];
            auto blurFft = new float[ n*n ];
            auto diff    = new float[ n*n ];

            srand(350471643);
            fillWithRandomValues( NULL, blurRaw, n*n );
            memcpy( blurFft, blurRaw, n*n*sizeof(float) );

            gaussianBlur   ( blurRaw, n,n, sigma );
            gaussianBlurFft( blurFft, n,n, sigma );

            #ifdef USE_PNG
                plotPng( blurRaw, n,n, "blurRaw.png" );
                plotPng( blurFft, n,n, "blurFft.png" );

                /* make and output difference map */
                for ( auto i = 0u; i < n*n; ++i )
                    diff[i] = blurRaw[i] - blurFft[i];
                plotPng( diff, n,n, "blurDiff.png" );
            #endif

            std::cout << "n=" << n << ", s=" << sigma << "\n";
            compareFloatArray( blurRaw, blurFft, n,n, sigma, __LINE__, 0.1, false /* print raw values on error */ );
            std::cout << "." << std::flush;

            delete[] diff;
            delete[] blurRaw;
            delete[] blurFft;
        }
        std::cout << " OK" << std::endl;
        #endif
    }

    TestGaussian::TestGaussian()
    {
        using namespace imresh::algorithms::cuda;
        using namespace imresh::libs;

        /* allocate pinned memory */
        mallocPinnedArray( &pData  , nMaxElements );
        mallocPinnedArray( &pResult, nMaxElements );
        mallocCudaArray( &dpData, nMaxElements );
        pResultCpu = new float[nMaxElements];
        pSolution  = new float[nMaxElements];
        /* fill test data with random numbers from [-0.5,0.5] */
        srand(350471643);
        fillWithRandomValues( dpData, pData, nMaxElements );
    }

    TestGaussian::~TestGaussian()
    {
        delete[] pResultCpu;
        delete[] pSolution;
        CUDA_ERROR( cudaFree( dpData ) );
        CUDA_ERROR( cudaFreeHost( pData ) );
        CUDA_ERROR( cudaFreeHost( pResult ) );
    }

    void TestGaussian::operator()( void )
    {
        testGaussianDiracDeltas();
        testGaussianRandomSingleData();
        testGaussianConstantValuesPerRowLine();
        testGaussianConstantValues();
        benchmarkGaussianGeneralRandomValues();
        testFourierConvolution();
    }



} // namespace algorithms
} // namespace imresh
