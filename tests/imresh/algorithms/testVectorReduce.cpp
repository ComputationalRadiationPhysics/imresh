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
#include <limits>
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
#include "libs/cudacommon.hpp"
#include "benchmarkHelper.hpp"


namespace imresh
{
namespace algorithms
{


    unsigned int constexpr nRepetitions = 20;


    template<class T_PREC>
    bool compareFloat( const char * file, int line, T_PREC a, T_PREC b, T_PREC marginFactor = 1.0 )
    {
        auto const max = std::max( std::abs(a), std::abs(b) );
        if ( max == 0 )
            return true; // both are 0 and therefore equal
        auto const relErr = fabs( a - b ) / max;
        auto const maxRelErr = marginFactor * std::numeric_limits<T_PREC>::epsilon();
        if ( not ( relErr <= maxRelErr ) )
            printf( "[%s:%i] relErr: %f > %f :maxRelErr!\n", file, line, relErr, maxRelErr );
        return relErr <= maxRelErr;
    }


    void testVectorReduce( void )
    {
        using namespace std::chrono;
        using namespace benchmark::imresh::algorithms::cuda;
        using namespace imresh::algorithms::cuda;
        using namespace imresh::algorithms;
        using namespace imresh::libs;

        const unsigned nMaxElements = 64*1024*1024;  // ~4000x4000 pixel
        auto pData = new float[nMaxElements];

        srand(350471643);
        for ( unsigned i = 0; i < nMaxElements; ++i )
            pData[i] = ( (float) rand() / RAND_MAX ) - 0.5f;
        float * dpData;
        mallocCudaArray( &dpData, nMaxElements );
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
        /* in order to filter out page time outs or similarily long random wait
         * times, we repeat the measurement nRepetitions times and choose the
         * shortest duration measured */

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        using clock = std::chrono::high_resolution_clock;

        std::cout << "vector length : cudaVectorMax (global atomic) | cudaVectorMax (global atomic) | cudaVectorMax (shared memory) | cudaVectorMax (shared memory+warp reduce) | cudaVectorMax (__shfl_down) | vectorMax | cudaVectorMin (__shfl_down) | vectorMin\n";
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

            #define TIME_GPU( FUNC, OBVIOUS_VALUE )                          \
            {                                                                \
                minTime = FLT_MAX;                                           \
                for ( unsigned iRepetition = 0; iRepetition < nRepetitions;  \
                      ++iRepetition )                                        \
                {                                                            \
                    cudaEventRecord( start );                                \
                    auto cudaReduced = FUNC( dpData, nElements );            \
                    cudaEventRecord( stop );                                 \
                    cudaEventSynchronize( stop );                            \
                    cudaEventElapsedTime( &milliseconds, start, stop );      \
                    minTime = fmin( minTime, milliseconds );                 \
                    assert( cudaReduced == OBVIOUS_VALUE );                  \
                }                                                            \
                std::cout << std::setw(8) << minTime << " |" << std::flush;  \
            }

            TIME_GPU( cudaVectorMaxGlobalAtomic2    , obviousMaximum )
            TIME_GPU( cudaVectorMaxGlobalAtomic     , obviousMaximum )
            TIME_GPU( cudaVectorMaxSharedMemory     , obviousMaximum )
            TIME_GPU( cudaVectorMaxSharedMemoryWarps, obviousMaximum )
            TIME_GPU( cudaVectorMax                 , obviousMaximum )

            /* time CPU */
            #define TIME_CPU( FUNC, OBVIOUS_VALUE )                          \
            {                                                                \
                minTime = FLT_MAX;                                           \
                for ( unsigned iRepetition = 0; iRepetition < nRepetitions;  \
                      ++iRepetition )                                        \
                {                                                            \
                    clock0 = clock::now();                                   \
                    auto cpuMax = FUNC( pData, nElements );                  \
                    clock1 = clock::now();                                   \
                    auto seconds = duration_cast<duration<float>>(           \
                                        clock1 - clock0 );                   \
                    minTime = fmin( minTime, seconds.count() * 1000 );       \
                    assert( cpuMax == OBVIOUS_VALUE );                       \
                }                                                            \
                std::cout << std::setw(8) << minTime << " |" << std::flush;  \
            }

            TIME_CPU( vectorMax, obviousMaximum )

            /* Minimum */
            pData[iObviousValuePos] = obviousMinimum;
            CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

            TIME_GPU( cudaVectorMin, obviousMinimum )
            TIME_CPU( vectorMin, obviousMinimum )

            /* set obvious value back to random value */
            pData[iObviousValuePos] = (float) rand() / RAND_MAX;
            std::cout << "\n";

            #undef TIME_GPU
            #undef TIME_CPU
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
        using namespace std::chrono;
        using namespace benchmark::imresh::algorithms::cuda;   // cudaCalculateHioErrorBitPacked
        using namespace imresh::algorithms::cuda;   // cudaKernelCalculateHioError
        using namespace imresh::libs;               // calculateHioError, mallocCudaArray
        using namespace imresh::tests;              // getLogSpacedSamplingPoints

        const unsigned nMaxElements = 64*1024*1024;  // ~4000x4000 pixel

        /* allocate */
        cufftComplex * dpData, * pData;
        unsigned char * dpIsMaskedChar, * pIsMaskedChar;
        float         * dpIsMasked    , * pIsMasked;
        unsigned      * dpBitMasked   , * pBitMasked;
        auto const nBitMaskedElements = ceilDiv( nMaxElements, 8 * sizeof( dpBitMasked[0] ) );
        mallocCudaArray( &dpIsMaskedChar, nMaxElements       );
        mallocCudaArray( &dpData        , nMaxElements       );
        mallocCudaArray( &dpIsMasked    , nMaxElements       );
        mallocCudaArray( &dpBitMasked   , nBitMaskedElements );
        pData         = new cufftComplex [ nMaxElements ];
        pIsMaskedChar = new unsigned char[ nMaxElements ];
        pIsMasked     = new float        [ nMaxElements ];
        pBitMasked    = new unsigned[ nBitMaskedElements ];
        /* allocate result buffer for reduced values of calculateHioError
         * kernel call */
        float nMaskedPixels, * dpnMaskedPixels;
        float totalError   , * dpTotalError;
        mallocCudaArray( &dpnMaskedPixels, 1 );
        mallocCudaArray( &dpTotalError   , 1 );

        /* initialize mask randomly */
        assert( sizeof(int) == 4 );
        srand(350471643);
        for ( auto i = 0u; i < nBitMaskedElements; ++i )
            pBitMasked[i] = rand() % UINT_MAX;
        unpackBitMask( pIsMasked, pBitMasked, nMaxElements );
        for ( auto i = 0u; i < nMaxElements; ++i )
        {
            pIsMaskedChar[i] = pIsMasked[i];
            assert( pIsMaskedChar[i] == 0 or pIsMaskedChar[i] == 1 );
        }

        std::cout << "[unpacked] ";
        for ( int i = 0; i < 32; ++i )
            std::cout << pIsMasked[i];
        std::cout << "\n";
        std::cout << "[  packed] " << std::bitset<32>( pBitMasked[0] ) << "\n";

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
        CUDA_ERROR( cudaMemcpy( dpIsMaskedChar, pIsMaskedChar, nMaxElements * sizeof( pIsMaskedChar[0] ), cudaMemcpyHostToDevice ) );

        std::cout << "test with randomly masked pythagorean triples";
        /* because the number of elements we include only increases the number
         * of found masked elements should also only increase. */
        float nLastMaskedPixels = 0;
        for ( auto nElements : getLogSpacedSamplingPoints( 2, nMaxElements, 50 ) )
        {
            std::cout << "." << std::flush;

            CUDA_ERROR( cudaMemset( dpnMaskedPixels, 0, sizeof(float) ) );
            CUDA_ERROR( cudaMemset( dpTotalError   , 0, sizeof(float) ) );
            cudaKernelCalculateHioError<<<3,256>>>
                ( dpData, dpIsMasked, nElements, false /* don't invert mask */,
                  dpTotalError, dpnMaskedPixels );
            CUDA_ERROR( cudaMemcpy( &nMaskedPixels, dpnMaskedPixels,
                                    sizeof(float), cudaMemcpyDeviceToHost) );
            CUDA_ERROR( cudaMemcpy( &totalError, dpTotalError,
                                    sizeof(float), cudaMemcpyDeviceToHost) );
            /* Calculation done, now check if everything is correct */
            if ( totalError < 16777216 ) // float vlaues higher round to multiple of 2
            {
                assert( nLastMaskedPixels <= nMaskedPixels );
                assert( (unsigned) totalError % 5 == 0 );
                assert( nMaskedPixels * 5 == totalError );
            }
            nLastMaskedPixels = nMaskedPixels;


            /* check char version */
            CUDA_ERROR( cudaMemset( dpnMaskedPixels, 0, sizeof(float) ) );
            CUDA_ERROR( cudaMemset( dpTotalError   , 0, sizeof(float) ) );
            cudaKernelCalculateHioError<<<3,256>>>
                ( dpData, dpIsMaskedChar, nElements, false /* don't invert mask */,
                  dpTotalError, dpnMaskedPixels );
            CUDA_ERROR( cudaMemcpy( &nMaskedPixels, dpnMaskedPixels,
                                    sizeof(float), cudaMemcpyDeviceToHost) );
            CUDA_ERROR( cudaMemcpy( &totalError, dpTotalError,
                                    sizeof(float), cudaMemcpyDeviceToHost) );
            /* Calculation done, now check if everything is correct */
            if ( totalError < 16777216 ) // float vlaues higher round to multiple of 2
            {
                assert( nLastMaskedPixels == nMaskedPixels );
                assert( (unsigned) totalError % 5 == 0 );
                assert( nMaskedPixels * 5 == totalError );
            }



            /* check packed bit version */
            CUDA_ERROR( cudaMemset( dpnMaskedPixels, 0, sizeof(float) ) );
            CUDA_ERROR( cudaMemset( dpTotalError   , 0, sizeof(float) ) );
            cudaKernelCalculateHioErrorBitPacked<<<1,32>>>
                ( dpData, dpBitMasked, nElements, dpTotalError, dpnMaskedPixels );
            CUDA_ERROR( cudaMemcpy( &nMaskedPixels, dpnMaskedPixels,
                                    sizeof(float), cudaMemcpyDeviceToHost) );
            CUDA_ERROR( cudaMemcpy( &totalError, dpTotalError,
                                    sizeof(float), cudaMemcpyDeviceToHost) );
            /* Calculation done, now check if everything is correct */
            if ( totalError < 16777216 ) // float vlaues higher round to multiple of 2
            {
                if ( not ( nLastMaskedPixels == nMaskedPixels ) )
                {
                    printf( "nLastMaskedPixels: %f, nMaskedPixels: %f, totalError: %f\n", nLastMaskedPixels, nMaskedPixels, totalError );
                    assert( nLastMaskedPixels == nMaskedPixels );
                }
                if ( not ( (unsigned) totalError % 5 == 0 ) )
                {
                    printf( "totalError: %f, nMaskedPixels: %f\n", totalError, nMaskedPixels );
                    assert( (unsigned) totalError % 5 == 0 );
                }
                assert( nMaskedPixels * 5 == totalError );
            }
            else
            {
                /* no use continuing this loop if we can't assert anything */
                break;
            }

            #ifdef USE_FFTW
                static_assert( sizeof( cufftComplex ) == sizeof( fftwf_complex ), "" );

                /* now compare with CPU version which should give the exact same
                 * result, as there should be no floating point rounding 4s
                 * for relatively short array ( < 1e6 ? ) */
                float nMaskedPixelsCpu, totalErrorCpu;
                calculateHioError( (fftwf_complex*) pData, pIsMasked, nElements, /* is inverted:  */ false, &totalErrorCpu, &nMaskedPixelsCpu );

                /* when rounding errors occur the order becomes important */
                if ( totalError < 16777216 )
                {
                    assert( compareFloat( __FILE__, __LINE__, totalError, totalErrorCpu, sqrtf(nElements) ) );
                    assert( nMaskedPixelsCpu == nMaskedPixels );
                }
            #endif
        }
        std::cout << "OK\n";

        /* benchmark with random numbers */

        for ( auto i = 0u; i < nBitMaskedElements; ++i )
        {
            pData[i].x = (float) rand() / RAND_MAX;
            pData[i].y = (float) rand() / RAND_MAX;
        }
        CUDA_ERROR( cudaMemcpy( dpData, pData, nMaxElements * sizeof( pData[0] ), cudaMemcpyHostToDevice ) );

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        using clock = std::chrono::high_resolution_clock;

        std::cout << "time in milliseconds:\n";
        std::cout << "vector length : cudaCalcHioError(uint32_t) | cudaCalcHioError(char) | cudaCalcHioError(packed) | calcHioError (CPU) |\n";
        for ( auto nElements : getLogSpacedSamplingPoints( 2, nMaxElements, 50 ) )
        {
            std::cout << std::setw(8) << nElements << " : ";
            float milliseconds, minTime;
            decltype( clock::now() ) clock0, clock1;

            {
                float error;
                #define TIME_GPU( FUNC, MASK )                              \
                minTime = FLT_MAX;                                          \
                for ( auto iRepetition = 0u; iRepetition < nRepetitions;    \
                      ++iRepetition )                                       \
                {                                                           \
                    cudaEventRecord( start );                               \
                    error = FUNC( dpData, MASK, nElements );                \
                    cudaEventRecord( stop );                                \
                    cudaEventSynchronize( stop );                           \
                    cudaEventElapsedTime( &milliseconds, start, stop );     \
                    minTime = fmin( minTime, milliseconds );                \
                    assert( error <= nElements );                           \
                }                                                           \
                std::cout << std::setw(8) << minTime << " |" << std::flush;

                TIME_GPU( cudaCalculateHioError, dpIsMasked )
                auto unpackedError = error;
                TIME_GPU( cudaCalculateHioError, dpIsMaskedChar ) // sets error
                compareFloat( __FILE__, __LINE__, unpackedError, error, sqrtf(nElements) );
                TIME_GPU( cudaCalculateHioErrorBitPacked, dpBitMasked ) // sets error
                compareFloat( __FILE__, __LINE__, unpackedError, error, sqrtf(nElements) );
            }
            #ifdef USE_FFTW
                /* time CPU */
                minTime = FLT_MAX;
                for ( auto iRepetition = 0u; iRepetition < nRepetitions;
                      ++iRepetition )
                {
                    clock0 = clock::now();
                    auto error = calculateHioError( (fftwf_complex*) pData, pIsMasked, nElements );
                    clock1 = clock::now();
                    auto seconds = duration_cast<duration<float>>( clock1 - clock0 );
                    minTime = fmin( minTime, seconds.count() * 1000 );
                    assert( error <= nElements );
                }
            #endif
            std::cout << std::setw(8) << minTime << "\n" << std::flush;
        }

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
