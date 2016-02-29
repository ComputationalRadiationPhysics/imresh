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
#include <chrono>
#include <vector>
#include <cmath>
#include <cfloat>    // FLT_MAX
#include <cuda_to_cupla.hpp>
#include "algorithms/vectorReduce.hpp"
#include "libs/cudacommon.hpp"
#include "benchmarkHelper.hpp"


namespace imresh
{
namespace algorithms
{

    void testVectorElementwise( void )
    {
#if false
        using namespace imresh::algorithms::cuda;
        using namespace imresh::algorithms;

        const unsigned nMaxElements = 64*1024*1024;  // ~4000x4000 pixel
        auto pData = new float[nMaxElements];

        srand(350471643);
        for ( unsigned i = 0; i < nMaxElements; ++i )
            pData[i] = ( (float) rand() / RAND_MAX ) - 0.5f;
        float * dpData;
        CUDA_ERROR( cudaMalloc( (void**)&dpData, nMaxElements*sizeof(dpData[0]) ) );
        CUDA_ERROR( cudaMemcpy( dpData, pData, nMaxElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

        /* Test for array of length 1 */
        assert( vectorMin( pData, 1 ) == pData[0] );
        assert( vectorMax( pData, 1 ) == pData[0] );
        assert( vectorSum( pData, 1 ) == pData[0] );
        assert( cudaVectorMin( dpData, 1 ) == pData[0] );
        assert( cudaVectorMax( dpData, 1 ) == pData[0] );
        assert( cudaVectorSum( dpData, 1 ) == pData[0] );

        /* do some checks with longer arrays and obvious results */
        float obviousMaximum = 7.37519;
        float obviousMinimum =-7.37519;
        const unsigned nRepetitions = 20;
        /* in order to filter out page time outs or similarily long random wait
         * times, we repeat the measurement nRepetitions times and choose the
         * shortest duration measured */

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        using clock = std::chrono::high_resolution_clock;

        std::cout << "vector length : cudaVectorMax (shared memory) | cudaVectorMax (shared memory+warp reduce) | cudaVectorMax (__shfl_down) | vectorMax | cudaVectorMin (__shfl_down) | vectorMin\n";
        using namespace imresh::tests;
        for ( auto nElements : getLogSpacedSamplingPoints( 2, nMaxElements, 50 ) )
        {
            std::cout << std::setw(8) << nElements << " : ";
            float milliseconds, minTime;
            decltype( clock::now() ) clock0, clock1;

            int iObviousValuePos = rand() % nElements;
            // std::cout << "iObviousValuePos = " << iObviousValuePos << "\n";
            // std::cout << "nElements        = " << nElements << "\n";


            /* Maximum */
            pData[iObviousValuePos] = obviousMaximum;
            CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

            /* time CUDA shared memory version */
            minTime = FLT_MAX;
            for ( unsigned iRepetition = 0; iRepetition < nRepetitions; ++iRepetition )
            {
                cudaEventRecord( start );
                auto cudaMax = cudaVectorMaxSharedMemory( dpData, nElements );
                cudaEventRecord( stop );
                cudaEventSynchronize( stop );
                cudaEventElapsedTime( &milliseconds, start, stop );
                minTime = fmin( minTime, milliseconds );
                assert( cudaMax == obviousMaximum );
            }
            std::cout << std::setw(8) << minTime << " |" << std::flush;

            /* time CUDA shared memory version + warp reduce */
            minTime = FLT_MAX;
            for ( unsigned iRepetition = 0; iRepetition < nRepetitions; ++iRepetition )
            {
                cudaEventRecord( start );
                auto cudaMax = cudaVectorMaxSharedMemoryWarps( dpData, nElements );
                cudaEventRecord( stop );
                cudaEventSynchronize( stop );
                cudaEventElapsedTime( &milliseconds, start, stop );
                minTime = fmin( minTime, milliseconds );
                assert( cudaMax == obviousMaximum );
            }
            std::cout << std::setw(8) << minTime << " |" << std::flush;

            /* time CUDA (warp reduce)*/
            minTime = FLT_MAX;
            for ( unsigned iRepetition = 0; iRepetition < nRepetitions; ++iRepetition )
            {
                cudaEventRecord( start );
                auto cudaMax = cudaVectorMax( dpData, nElements );
                cudaEventRecord( stop );
                cudaEventSynchronize( stop );
                cudaEventElapsedTime( &milliseconds, start, stop );
                minTime = fmin( minTime, milliseconds );
                assert( cudaMax == obviousMaximum );
            }
            std::cout << std::setw(8) << minTime << " |" << std::flush;

            /* time CPU */
            minTime = FLT_MAX;
            for ( unsigned iRepetition = 0; iRepetition < nRepetitions; ++iRepetition )
            {
                clock0 = clock::now();
                auto cpuMax = vectorMax( pData, nElements );
                clock1 = clock::now();
                milliseconds = ( clock1-clock0 ).count() / 1000.0;
                minTime = fmin( minTime, milliseconds );
                assert( cpuMax == obviousMaximum );
            }
            std::cout << std::setw(8) << minTime << " |" << std::flush;


            /* Minimum */
            pData[iObviousValuePos] = obviousMinimum;
            CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

            /* time CUDA */
            minTime = FLT_MAX;
            for ( unsigned iRepetition = 0; iRepetition < nRepetitions; ++iRepetition )
            {
                cudaEventRecord( start );
                auto cudaMin = cudaVectorMin( dpData, nElements );
                cudaEventRecord( stop );
                cudaEventSynchronize( stop );
                cudaEventElapsedTime( &milliseconds, start, stop );
                minTime = fmin( minTime, milliseconds );
                assert( cudaMin == obviousMinimum );
            }
            std::cout << std::setw(8) << minTime << " |" << std::flush;

            /* time CPU */
            minTime = FLT_MAX;
            for ( unsigned iRepetition = 0; iRepetition < nRepetitions; ++iRepetition )
            {
                clock0 = clock::now();
                auto cpuMin = vectorMin( pData, nElements );
                clock1 = clock::now();
                milliseconds = ( clock1-clock0 ).count() / 1000.0;
                minTime = fmin( minTime, milliseconds );
                assert( cpuMin == obviousMinimum );
            }
            std::cout << std::setw(8) << minTime << "\n" << std::flush;

            /* set obvious value back to random value */
            pData[iObviousValuePos] = (float) rand() / RAND_MAX;
        }


        //for ( unsigned nElements = 2; nElements

        CUDA_ERROR( cudaFree( dpData ) );
        delete[] pData;
#endif
    }


} // namespace algorithms
} // namespace imresh
