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


#include "testFft.hpp"


#include <cstdlib>              // srand, rand, RAND_MAX
#include <cstring>              // memcpy
#include <cmath>                // fmin, sqrt, floor
#include <chrono>               // high_resolution_clock
#include <iostream>             // cout
#include <iomanip>              // setw
#include <cassert>
#include <vector>
#include <cuda_to_cupla.hpp>    // cudaMemcpy

#include "libs/cufft_to_cupla.hpp"  // cufftComplex, FFT_Definition, ...
#ifdef USE_FFTW
#   include <fftw3.h>           // fftwf_plan_dft_2d, fftwf_complex, ...
#endif
#include "benchmarkHelper.hpp"  // getLogSpacedSamplingPoints
#include "libs/cudacommon.hpp"  // CUDA_ERROR, mallocCudaArray


namespace imresh
{
namespace algorithms
{


    /* in order to filter out page time outs or similarly long random wait
     * times, we repeat the measurement nRepetitions times and choose the
     * shortest duration measured */
    unsigned int constexpr nRepetitions = 20;
    unsigned int constexpr nMaxElements = 1024*1024;  // ~8000 x 8000 px


    void print2dArray
    (
        cufftComplex * const data,
        unsigned int const Nx,
        unsigned int const Ny
    )
    {
        for ( auto iy = 0u; iy < Ny; ++iy )
        {
            for ( auto ix = 0u; ix < Nx; ++ix )
            {
                std::cout << "( " << data[ iy * Nx + ix ].x << ", "
                                  << data[ iy * Nx + ix ].y << " ) ";
            }
            std::cout << std::endl;
        }
    }

    void testFftCheckerboard( void )
    {
        using imresh::libs::mallocCudaArray;

        /* allocate */
        unsigned int constexpr Nx = 4;
        unsigned int constexpr Ny = 4;
        auto pData = new cufftComplex[ Nx*Ny ];
        cufftComplex * dpData, *dpResult;
        mallocCudaArray( &dpData  , Nx*Ny );
        mallocCudaArray( &dpResult, Nx*Ny );

        /* create plan and wrap data */
        using GpuFftPlanFwd = FFT_Definition<
            FFT_Kind::Complex2Complex,
            2 /* dims */,
            float,
            std::true_type /* forward */,
            false /* not in-place */
        >;
        auto dpDataWrapped   = GpuFftPlanFwd::wrapInput ( wrapComplexDevicePointer( dpData  , Ny, Nx ) );
        auto dpResultWrapped = GpuFftPlanFwd::wrapOutput( wrapComplexDevicePointer( dpResult, Ny, Nx ) );

        /* initialize */
        for ( auto i = 0u; i < Nx*Ny/2; ++i )
        {
            pData[ 2*i+0 ].x = 1;
            pData[ 2*i+0 ].y = 0;
            pData[ 2*i+1 ].x = 0;
            pData[ 2*i+1 ].y = 0;
        }
        print2dArray( pData, Nx, Ny );
        std::cout << std::endl;
        CUDA_ERROR( cudaMemcpy( dpData, pData, Nx*Ny * sizeof( dpData[0] ), cudaMemcpyHostToDevice ) );

        auto fftForward = makeFftPlan( dpDataWrapped, dpResultWrapped );
        fftForward( dpDataWrapped, dpResultWrapped );

        CUDA_ERROR( cudaMemcpy( pData, dpResult, Nx*Ny * sizeof( dpData[0] ), cudaMemcpyDeviceToHost ) );
        print2dArray( pData, Nx, Ny );

        CUDA_ERROR( cudaFree( dpData  ) );
        CUDA_ERROR( cudaFree( dpResult) );
    }

    void testFft( void )
    {
        testFftCheckerboard();

        using namespace std::chrono;
        using namespace imresh::algorithms;
        using imresh::libs::mallocCudaArray;
        using namespace imresh::tests;

        auto pData = new cufftComplex[ nMaxElements ];
        cufftComplex * dpData, *dpResult;
        for ( auto i = 0u; i < nMaxElements; ++i )
        {
            pData[i].x = (float) rand() / RAND_MAX;
            pData[i].y = (float) rand() / RAND_MAX;
        }
        mallocCudaArray( &dpData  , nMaxElements );
        mallocCudaArray( &dpResult, nMaxElements );
        CUDA_ERROR( cudaMemcpy( dpData, pData, nMaxElements * sizeof( dpData[0] ), cudaMemcpyHostToDevice ) );

        #ifdef USE_FFTW
            auto pResult = new fftwf_complex[ nMaxElements ];
        #endif

        std::cout << "\n";
        std::cout << "FFT comparison timings in milliseconds:\n";
        std::cout << " image size   :                          |\n";
        std::cout << "(nCols,nRows) :       HaLT / LiFFT       |";
                  /* "(    1,    1) :  0.0001302 +-   0.000385 | "*/
        #ifdef USE_FFTW
            std::cout << std::setw(17) << "FFTW3";
        #endif
        /*#ifdef USE_CUFFT
            std::cout << std::setw(26) << "cuFFT";
        #endif*/
        std::cout << std::endl;

        for ( auto nElements : getLogSpacedSamplingPoints( 2, nMaxElements, 20 ) )
        {
            unsigned int const Nx  = floor(sqrt( nElements ));
            unsigned int const Ny = Nx;
            assert( Nx*Ny<= nMaxElements );
            nElements = Nx * Ny;

            std::cout << "(" << std::setw(5) << Nx << ","
                             << std::setw(5) << Ny << ") : ";

            #define TIME_COMMAND( COMMAND )                                    \
            {                                                                  \
                std::vector<float> times;                                      \
                for ( auto iRep = 0u; iRep < nRepetitions; ++iRep )            \
                {                                                              \
                    using clock = std::chrono::high_resolution_clock;          \
                    auto const t0 = clock::now();                              \
                        COMMAND;                                               \
                    auto const t1 = clock::now();                              \
                    auto const seconds = duration_cast< duration<float> >      \
                                         ( t1-t0 );                            \
                    times.push_back( seconds.count() * 1000 );                 \
                }                                                              \
                std::cout << std::setprecision(4)                              \
                          << std::setw(10) << mean  ( times ) << " +- "        \
                          << std::setw(10) << stddev( times ) << " | ";        \
            }

            /* LiFFT */
            using GpuFftPlanFwd = FFT_Definition<
                FFT_Kind::Complex2Complex,
                2 /* dims */,
                float,
                std::true_type /* forward */,
                false /* not in-place */
            >;
            auto inputData  = GpuFftPlanFwd::wrapInput ( wrapComplexDevicePointer( dpData  , Ny, Nx ) );
            auto outputData = GpuFftPlanFwd::wrapOutput( wrapComplexDevicePointer( dpResult, Ny, Nx ) );
            auto fftForward = makeFftPlan( inputData, outputData );
            TIME_COMMAND( fftForward( inputData, outputData ) )

            /* FFTW */
            #ifdef USE_FFTW
                auto cpuFtPlan = fftwf_plan_dft_2d( Ny, Nx, (fftwf_complex*) pData, pResult, FFTW_FORWARD, FFTW_ESTIMATE );

                TIME_COMMAND( fftwf_execute( cpuFtPlan ) )

                fftwf_destroy_plan( cpuFtPlan );
            #endif

            /* cuFFT (only sketched out, @todo, @see 074308e3da9755f8c8) */
            /* #ifdef USE_CUFFT
                cufftHandle gpuFtPlan;
                CUFFT_ERROR( cufftPlan2d( &gpuFtPlan, Ny, Nx, CUFFT_C2C ) );

                TIME_COMMAND( CUFFT_ERROR( cufftExecC2C( gpuFtPlan, dpData, dpResult, CUFFT_FORWARD ) ) );

                CUFFT_ERROR( cufftDestroy( gpuFtPlan ) );
            #endif */

            std::cout << std::endl;
        }

        CUDA_ERROR( cudaFree( dpData  ) );
        CUDA_ERROR( cudaFree( dpResult) );
        delete[] pData;
        #ifdef USE_FFTW
            delete[] pResult;
        #endif
    }


    /* compare a naive fourier transform with actually used one */

    #ifndef M_PI
    #   define M_PI 3.141592653589793238462643383279502884
    #endif

    /**
     * Calculates fourier coefficients a_k and b_k up for k=1 to k=rnCoefficients
     *
     * @param[in]  f continuous function to transform
     * @param[in]  rnCoefficients number of complex(!) coefficients to calculate,
     *             the returned float array will contain 2*rnCoefficients elements!
     * @param[in]  sign -1 for forward FT and +1 for backward FT
     * @param[out] rCoefficients pointer to allocated float array which will
     *             rnCoefficients cosine, i.e. real, fourier coefficients, followed
     *             by nCoefficients sine, i.e. imaginary, fourier coefficeints
     **/
    void dcft
    (
        unsigned int const rnValues,
        float*       const rFunc   ,
        float*       const rResult ,
        int          const sign
    )
    {
        /* F(k_n) = \sum_j f(x_j) exp(i*k_n*x_j)
         *          with k_n = 2*pi*n/N and x_j = j/N
         *        = \sum_j f_j exp( 2*pi*i*n*j/N ) */
        for ( auto n = 0u; n < rnValues; ++n )
        {
            rResult[ 2*n+0 ] = 0;
            rResult[ 2*n+1 ] = 0;
            for ( auto j = 0u; j < rnValues; ++j )
            {
                rResult[ 2*n+0 ] += rFunc[ 2*j+0 ] * std::cos( 2 * M_PI * sign * n*j/rnValues )
                                  - rFunc[ 2*j+1 ] * std::sin( 2 * M_PI * sign * n*j/rnValues );
                rResult[ 2*n+1 ] += rFunc[ 2*j+0 ] * std::sin( 2 * M_PI * sign * n*j/rnValues )
                                  + rFunc[ 2*j+1 ] * std::cos( 2 * M_PI * sign * n*j/rnValues );
            }
        }
    }


    void testCompareDcftArray( std::vector< float > data )
    {
        assert( data.size() % 2 == 0 );
        unsigned int const N = data.size() / 2;

        /* malloc arrays needed */
        using imresh::libs::mallocCudaArray;
        cufftComplex * dpResult;
        mallocCudaArray( &dpResult, N );
        CUDA_ERROR( cudaMemcpy( dpResult, &data[0], data.size() * sizeof( data[0] ), cudaMemcpyHostToDevice ) );
        std::vector<cufftComplex> resultDcft ( N ),
                                  resultLifft( N );
        #ifdef USE_FFTW
        std::vector<cufftComplex> resultFftw ( N );
        #endif


        /* naively */
        dcft( N, (float*) &data[0], (float*) &resultDcft[0], -1 /* forward */ );

        /* LiFFT */
        using GpuFftPlanFwd = FFT_Definition<
            FFT_Kind::Complex2Complex,
            1,              /* dims */
            float,          /* used precision */
            std::true_type, /* forward */
            true            /* in-place */
        >;
        auto ioData = GpuFftPlanFwd::wrapInput( wrapComplexDevicePointer( dpResult, N ) );
        auto fftForward = makeFftPlan( ioData );
        fftForward( ioData );
        CUDA_ERROR( cudaMemcpy( &resultLifft[0], dpResult, N * sizeof( dpResult[0] ), cudaMemcpyDeviceToHost ) );

        /* FFTW */
        #ifdef USE_FFTW
            auto cpuFtPlan = fftwf_plan_dft_1d( N,
                (fftwf_complex*) &data[0], (fftwf_complex*) &resultFftw[0],
                FFTW_FORWARD, FFTW_ESTIMATE );
            fftwf_execute( cpuFtPlan );
            fftwf_destroy_plan( cpuFtPlan );
        #endif

        #define PRINT_COMPLEX_ARRAY(X)                                  \
        for ( auto i = 0u; i < N; ++i )                   \
            std::cout << "  (" << std::setprecision(5) << (X)[2*i]      \
                      << ","   << std::setprecision(5) << (X)[2*i+1]    \
                      << ")" << std::endl;
        std::cout << "Input:" << std::endl;
        PRINT_COMPLEX_ARRAY( data )
        std::cout << "Outputs:" << std::endl;
        std::cout << "        naive        |        LiFFT        |        FFTW         |" << std::endl;
        for ( auto i = 0u; i < N; ++i )
        {
            std::cout << std::setprecision(3)
                      << " (" << std::setw(8) << resultDcft[i].x
                      << ","  << std::setw(8) << resultDcft[i].y
                      << ") |"
                      << " (" << std::setw(8) << resultLifft[i].x
                      << ","  << std::setw(8) << resultLifft[i].y
                      << ") |"
            #ifdef USE_FFTW
                      << " (" << std::setw(8) << resultFftw[i].x
                      << ","  << std::setw(8) << resultFftw[i].y
                      << ") |"
            #endif
                      << std::endl;
        }
        std::cout << std::endl;
    }

    void testCompareDcftRealArray( std::vector< float > data )
    {
        std::vector< float > input( data.size() * 2 );
        for ( auto i = 0u; i < data.size(); ++i )
        {
            input[ 2*i + 0 ] = data[i]; /* Re */
            input[ 2*i + 1 ] = 0;       /* Im */
        }
        testCompareDcftArray( input );
    }

    template< class T_Func >
    void testCompareDcftFunction( T_Func f )
    {
        for ( auto nSamples : std::vector<int>{ 10, 100, 1024 } )
        {
            std::vector< float >( 2*nSamples );
            for ( int i = 0; i < nSamples; ++i )
            {
                /* initialize array @todo */
            }
        }
    }

    void testCompareDcft( void )
    {
        testCompareDcftRealArray( { 1 } );
        testCompareDcftRealArray( { 1, 0 } );
        testCompareDcftRealArray( { 1,0,1,0,1 } );
        testCompareDcftRealArray( { 1,0,1,0,1,0,1,0,1 } );

        #define L(X) []( float x ){ return X; }
        testCompareDcftFunction( L( std::sin( 1*x )) );
        testCompareDcftFunction( L( std::sin( 2*x )) );
        testCompareDcftFunction( L( std::sin( 3*x )) );
        testCompareDcftFunction( L( std::abs( x )  ) );
        testCompareDcftFunction( L( x > 0 ? 1 : 0  ) );
        #undef L
    }


} // namespace algorithms
} // namespace imresh
