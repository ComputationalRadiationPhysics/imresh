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


#include "testVectorReduce.hpp"

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstdlib>          // srand, rand
#include <cstdint>          // uint32_t, uint64_t
#include <chrono>
#include <vector>
#include <cmath>
#include <cfloat>           // FLT_MAX
#include <bitset>
#include <cuda_runtime.h>
#include <cufft.h>          // cufftComplex
#ifdef USE_FFTW
#   include <fftw3.h>
#   include "libs/hybridInputOutput.hpp"
#endif
#include "algorithms/vectorReduce.hpp"
#include "algorithms/cuda/cudaVectorReduce.hpp"
#include "benchmark/imresh/algorithms/cuda/cudaVectorReduce.hpp"
#include "libs/cudacommon.h"
#include "benchmarkHelper.hpp"


namespace imresh
{
namespace algorithms
{


    void testVectorReduce( void )
    {
        using namespace benchmark::imresh::algorithms::cuda;
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
    }



    template<class T_MASK, class T_PACKED>
    __attribute__(( optimize("unroll-loops") ))
    void unpackBitMask
    (
        T_MASK         * const __restrict__ rMask,
        T_PACKED const * const __restrict__ rPackedBits,
        unsigned int const nElements
    )
    {
        auto const nElem = rMask + nElements;
        auto constexpr nBits = sizeof( T_PACKED ) * 8u;
        auto iPacked = rPackedBits;

        for ( auto iElem = rMask; iElem < nElem; ++iPacked )
        {
            auto bitMask = T_PACKED(0x01) << ( nBits-1 );

            for ( auto iBit = 0u; iBit < nBits; ++iBit, ++iElem )
            {
                if ( iElem >= nElem )
                    break;

                assert( bitMask != T_MASK(0) );
                assert( iElem < rMask + nElements );
                assert( iPacked < rPackedBits + ceilDiv( nElements, nBits ) );

                *iElem = T_MASK( (*iPacked & bitMask) != 0 );
                bitMask >>= 1;
            }
        }
    }

    void testUnpackBitMask( void )
    {
        uint32_t packed = 0x33333333;
        constexpr auto nElements = 8 * sizeof( packed );
        bool unpacked[ nElements ];
        unpacked[ nElements-2 ] = 1;
        unpacked[ nElements-1 ] = 0;
        unpackBitMask( unpacked, &packed, nElements-2 );

        for ( auto i = 0u; i < (nElements-2)/2; ++i )
        {
            assert( unpacked[2*i+0] == i % 2 );
            assert( unpacked[2*i+1] == i % 2 );
        }
        assert( unpacked[ nElements-2 ] == 1 );
        assert( unpacked[ nElements-1 ] == 0 );
    }

    void testCalculateHioError( void )
    {
        using namespace imresh::algorithms::cuda;   // cudaKernelCalculateHioError
        using namespace imresh::libs;               // calculateHioError
        using namespace imresh::tests;              // getLogSpacedSamplingPoints

        const unsigned nMaxElements = 64*1024*1024;  // ~4000x4000 pixel

        /* allocate */
        cufftComplex * dpData, * pData;
        float    * dpIsMasked , * pIsMasked;
        unsigned * dpBitMasked, * pBitMasked;
        auto const nBitMaskedElements = ceilDiv( nMaxElements, 8 * sizeof( dpBitMasked[0] ) );
        CUDA_ERROR( cudaMalloc( &dpData     , nMaxElements * sizeof( dpData    [0] ) ) );
        CUDA_ERROR( cudaMalloc( &dpIsMasked , nMaxElements * sizeof( dpIsMasked[0] ) ) );
        CUDA_ERROR( cudaMalloc( &dpBitMasked, nBitMaskedElements * sizeof( dpBitMasked[0] ) ) );
        pData      = new cufftComplex[ nMaxElements ];
        pIsMasked  = new float[ nMaxElements ];
        pBitMasked = new unsigned[ nBitMaskedElements ];
        /* allocate result buffer for reduced values of calculateHioError
         * kernel call */
        float nMaskedPixels, * dpnMaskedPixels;
        float totalError   , * dpTotalError;
        CUDA_ERROR( cudaMalloc( &dpnMaskedPixels, sizeof(float) ) );
        CUDA_ERROR( cudaMalloc( &dpTotalError   , sizeof(float) ) );

        /* initialize mask randomly */
        assert( sizeof(int) == 4 );
        srand(350471643);
        for ( auto i = 0u; i < nBitMaskedElements; ++i )
            pBitMasked[i] = rand() % UINT_MAX;
        unpackBitMask( pIsMasked, pBitMasked, nMaxElements );

        /* initialize data with Pythagorean triple 3*3 + 4*4 = 5*5 for masked bits */
        for ( auto i = 0u; i < nMaxElements; ++i )
        {
            if ( pIsMasked[i] )
            {
                pData[i].x = 3.0f;
                pData[i].y = 4.0f;
            }
            else
            {
                pData[i].x = (float) rand() / RAND_MAX;
                pData[i].y = (float) rand() / RAND_MAX;
            }
        }
        /* if calculateHioError works correctly then we simply get
         * #masked * 5 as the mean complex norm error */

        /* push to GPU */
        CUDA_ERROR( cudaMemcpy( dpData     , pData     , nMaxElements * sizeof( pData    [0] ), cudaMemcpyHostToDevice ) );
        CUDA_ERROR( cudaMemcpy( dpIsMasked , pIsMasked , nMaxElements * sizeof( pIsMasked[0] ), cudaMemcpyHostToDevice ) );
        CUDA_ERROR( cudaMemcpy( dpBitMasked, pBitMasked, nBitMaskedElements * sizeof( pBitMasked[0] ), cudaMemcpyHostToDevice ) );

        std::cout << "test with randomly masked pythagorean triples";
        /* because the number of elements we include only increases the number
         * of found masked elements should also only increase. */
        float nLastMaskedPixels = 0;
        for ( auto nElements : getLogSpacedSamplingPoints( 2, nMaxElements, 50 ) )
        {
            std::cout << "." << std::flush;
            CUDA_ERROR( cudaMemset( dpnMaskedPixels, 0, sizeof(float) ) );
            CUDA_ERROR( cudaMemset( dpTotalError   , 0, sizeof(float) ) );
            cudaKernelCalculateHioError<<<1,256>>>
                ( dpData, dpIsMasked, nElements, false /* don't invert mask */,
                  dpTotalError, dpnMaskedPixels );
            CUDA_ERROR( cudaMemcpy( &nMaskedPixels, dpnMaskedPixels,
                                    sizeof(float), cudaMemcpyDeviceToHost) );
            CUDA_ERROR( cudaMemcpy( &totalError, dpTotalError,
                                    sizeof(float), cudaMemcpyDeviceToHost) );

            /* Calculation done, now check if everything is correct */
            assert( nLastMaskedPixels <= nMaskedPixels );
            assert( (unsigned) totalError % 5 == 0 );
            printf( "%u, %f\n", nMaskedPixels, totalError );
            assert( nMaskedPixels / 5 == totalError );

            nLastMaskedPixels = nMaskedPixels;

            #ifdef USE_FFTW
                static_assert( sizeof( cufftComplex ) == sizeof( fftwf_complex ), "" );

                /* now compare with CPU version which should give the exact same
                 * result, as there should be no floating point rounding errors
                 * for relatively short array ( < 1e6 ? ) */
                float nMaskedPixelsCpu, totalErrorCpu;
                calculateHioError( (fftwf_complex*) pData, pIsMasked, nElements, /* is inverted:  */ false, &totalErrorCpu, &nMaskedPixelsCpu );

                assert( totalErrorCpu == totalError );
                assert( nMaskedPixelsCpu == nMaskedPixels );
            #endif
        }
        std::cout << "OK\n";

#if false
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        using clock = std::chrono::high_resolution_clock;

        std::cout << "vector length : cudaVectorMax (shared memory) | cudaVectorMax (shared memory+warp reduce) | cudaVectorMax (__shfl_down) | vectorMax | cudaVectorMin (__shfl_down) | vectorMin\n";
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

#endif

        /* free */
        CUDA_ERROR( cudaFree( dpnMaskedPixels ) );
        CUDA_ERROR( cudaFree( dpTotalError    ) );
        CUDA_ERROR( cudaFree( dpData          ) );
        CUDA_ERROR( cudaFree( dpIsMasked      ) );
        CUDA_ERROR( cudaFree( dpBitMasked     ) );
        delete[] pData;
        delete[] pIsMasked;
        delete[] pBitMasked;
    }


} // namespace algorithms
} // namespace imresh
