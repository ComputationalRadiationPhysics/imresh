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


#include "testShrinkWrap.hpp"


#include <cstdlib>              // srand, rand
#include <cstring>              // memcpy
#include <cfloat>               // FLT_MAX
#include <cmath>                // fmin
#include <chrono>               // high_resolution_clock
#include <iostream>
#include <iomanip>              // setw
#include <cassert>
#include <vector>
#include <cuda_runtime_api.h>   // cudaMalloc, cudaMemcpy, cudaEventRecord, ...
#include <cufft.h>
#ifdef USE_FFTW
#   include <fftw3.h>
#endif
#include "algorithms/shrinkWrap.hpp"
#include "algorithms/cuda/cudaShrinkWrap.h"
#include "io/taskQueue.hpp"
#include "benchmarkHelper.hpp"  // getLogSpacedSamplingPoints
#include "examples/createTestData/createAtomCluster.hpp"
#include "libs/diffractionIntensity.hpp"
#include "libs/checkCufftError.hpp"
#include "libs/cudacommon.h"


namespace imresh
{
namespace algorithms
{


    /* in order to filter out page time outs or similarly long random wait
     * times, we repeat the measurement nRepetitions times and choose the
     * shortest duration measured */
    unsigned int constexpr nRepetitions = 20;
    unsigned int constexpr nMaxElements = 1024*1024;  // ~8000 x 8000 px
    unsigned int constexpr nShrinkWrapCycles = 4;


    void testShrinkWrap( void )
    {
        using namespace std::chrono;
        using namespace imresh::algorithms;
        using namespace imresh::algorithms::cuda;
        using examples::createTestData::createAtomCluster;
        using imresh::libs::diffractionIntensity;
        using namespace imresh::tests;  // mean, stddef, getLogSpacedSamplingPoints

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        using clock = std::chrono::high_resolution_clock;
        decltype( clock::now() ) clock0, clock1;
        duration<double> seconds;

        std::cout << "\n";
        std::cout << "shrink wrap comparison timings in milliseconds:\n";
        std::cout << " image size   :\n";
        std::cout << "(nCols,nRows) : cudaMalloc | cudaMemcpy | shrinkWrap (CPU) | cudaShrinkWrap | cudaFree |" << std::endl;

        for ( auto nElements : getLogSpacedSamplingPoints( 2, nMaxElements, 20 ) )
        {
            unsigned int const Nx  = floor(sqrt( nElements ));
            unsigned int const Ny = Nx;
            assert( Nx*Ny<= nMaxElements );
            nElements = Nx * Ny;

            auto pData = createAtomCluster( Nx, Ny );
            diffractionIntensity( pData, Nx, Ny );
            float * dpData;

            std::cout << "(" << std::setw(5) << Nx << ","
                             << std::setw(5) << Ny << ") : ";

            std::vector<float> timeCudaMalloc, timeCudaMemcpy, timeShrinkWrap, timeCudaShrinkWrap, timeCudaFree, timeCudaParallel;
            /* shrink wrap takes quite a bit, so only do one quarter of repetitions */
            for ( auto iRepetition = 0u; iRepetition < nRepetitions / 2;
                  ++iRepetition )
            {
                /* cudaMalloc */
                clock0 = clock::now();
                    CUDA_ERROR( cudaMalloc( (void**) &dpData, Nx*Ny*sizeof( dpData[0] ) ) );
                clock1 = clock::now();
                seconds = duration_cast<duration<double>>( clock1 - clock0 );
                timeCudaMalloc.push_back( seconds.count() * 1000 );

                /* cudaMemcpy */
                clock0 = clock::now();
                    CUDA_ERROR( cudaMemcpy( dpData, pData, Nx*Ny*sizeof( dpData[0] ), cudaMemcpyHostToDevice ) );
                clock1 = clock::now();
                seconds = duration_cast<duration<double>>( clock1 - clock0 );
                timeCudaMemcpy.push_back( seconds.count() * 1000 );

                /* shrinkWrap */
                clock0 = clock::now();
                #ifdef USE_FFTW
                    shrinkWrap( pData, Nx, Ny, nShrinkWrapCycles, FLT_MIN ); /* use FLT_MIN to specifiy it should never end or onfly after 32 iterations */
                #endif
                clock1 = clock::now();
                seconds = duration_cast<duration<double>>( clock1 - clock0 );
                timeShrinkWrap.push_back( seconds.count() * 1000 );

                /* cudaShrinkWrap */
                cudaEventRecord( start );
                    cudaShrinkWrap( pData, Nx, Ny, cudaStream_t(0), 12288/256, 256, nShrinkWrapCycles, FLT_MIN );
                cudaEventRecord( stop );
                cudaEventSynchronize( stop );
                float milliseconds;
                cudaEventElapsedTime( &milliseconds, start, stop );
                timeCudaShrinkWrap.push_back( milliseconds );

                /* cudaFree */
                clock0 = clock::now();
                    CUDA_ERROR( cudaFree( dpData ) );
                clock1 = clock::now();
                seconds = duration_cast<duration<double>>( clock1 - clock0 );
                timeCudaFree.push_back( seconds.count() * 1000 );

                /* multithreaded test */
                unsigned int const nConcurrentTasks = 12;
                auto tmpDataArray = new float[ Nx*Ny*nConcurrentTasks ];
                for ( auto i = 0u; i < nConcurrentTasks; ++i )
                    memcpy( tmpDataArray + i*Nx*Ny, pData, Nx*Ny * sizeof( pData[0] ) );
                clock0 = clock::now();
                {
                    using ImageDim = std::pair<unsigned int,unsigned int>;
                    imresh::io::taskQueueInit( );
                    for( auto i = 0u; i < nConcurrentTasks; i++ )
                    {
                        imresh::io::addTask( tmpDataArray + i*Nx*Ny, ImageDim{ Nx, Ny },
                                             []( float * a, ImageDim const b, std::string const c ){},
                                             "", nShrinkWrapCycles );
                    }
                    imresh::io::taskQueueDeinit( ); // synchronizes device
                }
                clock1 = clock::now();
                seconds = duration_cast<duration<double>>( clock1 - clock0 );
                timeCudaParallel.push_back( seconds.count() * 1000 / nConcurrentTasks );
                delete[] tmpDataArray;

            }
            std::cout << std::setprecision(4);
            std::cout << std::setw(8) << mean(timeCudaMalloc    ) << " +- " << std::setw(9) << stddev( timeCudaMalloc    ) << " | ";
            std::cout << std::setw(8) << mean(timeCudaMemcpy    ) << " +- " << std::setw(9) << stddev( timeCudaMemcpy    ) << " | ";
            std::cout << std::setw(8) << mean(timeShrinkWrap    ) << " +- " << std::setw(9) << stddev( timeShrinkWrap    ) << " | ";
            std::cout << std::setw(8) << mean(timeCudaParallel  ) << " +- " << std::setw(9) << stddev( timeCudaParallel  ) << " | ";
            std::cout << std::setw(8) << mean(timeCudaFree      ) << " +- " << std::setw(9) << stddev( timeCudaFree      ) << " | ";
            std::cout << std::setw(8) << mean(timeCudaShrinkWrap) << " +- " << std::setw(9) << stddev( timeCudaShrinkWrap) << " | ";
            std::cout << std::endl;
            /*
            std::cout << "timeShrinkWrap = ";
            for ( auto const & elem : timeShrinkWrap )
                std::cout << elem << " ";
            std::cout << "\n";
            */

            delete[] pData;
        }
    }


    void testFft( void )
    {
        using namespace std::chrono;
        using namespace imresh::algorithms;
        using namespace imresh::algorithms::cuda;
        using examples::createTestData::createAtomCluster;
        using imresh::libs::diffractionIntensity;
        using namespace imresh::tests;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        using clock = std::chrono::high_resolution_clock;
        decltype( clock::now() ) clock0, clock1;
        duration<double> seconds;

        auto pData = new cufftComplex[ nMaxElements ];
        cufftComplex * dpData, *dpResult;
        for ( auto i = 0u; i < nMaxElements; ++i )
        {
            pData[i].x = (float) rand() / RAND_MAX;
            pData[i].y = (float) rand() / RAND_MAX;
        }
        CUDA_ERROR( cudaMalloc( (void**) &dpData  , nMaxElements * sizeof( dpData  [0] ) ) );
        CUDA_ERROR( cudaMalloc( (void**) &dpResult, nMaxElements * sizeof( dpResult[0] ) ) );
        CUDA_ERROR( cudaMemcpy( dpData, pData,     nMaxElements * sizeof( dpData  [0] ), cudaMemcpyHostToDevice ) );

        #ifdef USE_FFTW
            auto pResult = new fftwf_complex[ nMaxElements ];
        #endif
        cufftHandle gpuFtPlan;

        std::cout << "\n";
        std::cout << "FFT comparison timings in milliseconds:\n";
        std::cout << " image size   :\n";
        std::cout << "(nCols,nRows) : fftw3 (CPU) | cuFFT |" << std::endl;

        for ( auto nElements : getLogSpacedSamplingPoints( 2, nMaxElements, 20 ) )
        {
            unsigned int const Nx  = floor(sqrt( nElements ));
            unsigned int const Ny = Nx;
            assert( Nx*Ny<= nMaxElements );
            nElements = Nx * Ny;

            #ifdef USE_FFTW
                auto cpuFtPlan = fftwf_plan_dft_2d( Ny, Nx, (fftwf_complex*) pData, pResult, FFTW_FORWARD, FFTW_ESTIMATE );
            #endif
            CUFFT_ERROR( cufftPlan2d( &gpuFtPlan, Ny /* nRows */, Nx /* nColumns */, CUFFT_C2C ) );

            std::cout << "(" << std::setw(5) << Nx << ","
                             << std::setw(5) << Ny << ") : ";

            std::vector<float> timeCpuFft, timeGpuFft;
            for ( auto iRepetition = 0u; iRepetition < nRepetitions;
                  ++iRepetition )
            {
                #ifdef USE_FFTW
                    /* fftw */
                    clock0 = clock::now();
                        fftwf_execute( cpuFtPlan );
                    clock1 = clock::now();
                    seconds = duration_cast<duration<double>>( clock1 - clock0 );
                    timeCpuFft.push_back( seconds.count() * 1000 );
                #endif

                /* cuFFT */
                cudaEventRecord( start );
                    CUFFT_ERROR( cufftExecC2C( gpuFtPlan, dpData, dpResult, CUFFT_FORWARD ) );
                cudaEventRecord( stop );
                cudaEventSynchronize( stop );
                float milliseconds;
                cudaEventElapsedTime( &milliseconds, start, stop );
                timeGpuFft.push_back( milliseconds );
            }
            std::cout << std::setprecision(4);
            std::cout << std::setw(10) << mean( timeCpuFft ) << " +- " << std::setw(10) << stddev( timeCpuFft ) << " | ";
            std::cout << std::setw(10) << mean( timeGpuFft ) << " +- " << std::setw(10) << stddev( timeGpuFft ) << " | ";
            std::cout << std::endl;

            CUFFT_ERROR( cufftDestroy( gpuFtPlan ) );
            #ifdef USE_FFTW
                fftwf_destroy_plan( cpuFtPlan );
            #endif
        }

        CUDA_ERROR( cudaFree( dpData  ) );
        CUDA_ERROR( cudaFree( dpResult) );
        delete[] pData;
        #ifdef USE_FFTW
            delete[] pResult;
        #endif
    }


} // namespace algorithms
} // namespace imresh
