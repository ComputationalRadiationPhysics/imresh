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


#include <cstring>              // memcpy
#include <cfloat>               // FLT_MAX, FLT_MIN
#include <cmath>                // sqrt, floor
#include <chrono>               // high_resolution_clock
#include <iostream>             // cout
#include <iomanip>              // setw
#include <cassert>
#include <vector>
#include <cuda_to_cupla.hpp>    // cudaMalloc, cudaMemcpy, cudaEventRecord, ...

#include "algorithms/shrinkWrap.hpp"
#include "algorithms/cuda/cudaShrinkWrap.hpp"
#include "io/taskQueue.hpp"
#include "benchmarkHelper.hpp"  // getLogSpacedSamplingPoints
#include "examples/createTestData/createAtomCluster.hpp"
#include "libs/diffractionIntensity.hpp"
#include "libs/cudacommon.hpp"  // CUDA_ERROR


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


    void testShrinkWrapExampleImages( void )
    {
        /**
         * test in and out of examples also seen in outputCreation
         * also add some fuzzing, like:
         *   - shifting the center
         *   - cropping
         *   - extending
         *   - blacking out border values
         *   - whitening out center pixels
         *   - scaling
         *   - distortions if possible
         *   - noise
         **/
    }


    void testShrinkWrap( void )
    {
        testShrinkWrapExampleImages();

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
                seconds = duration_cast<duration<float>>( clock1 - clock0 );
                timeCudaMalloc.push_back( seconds.count() * 1000 );

                /* cudaMemcpy */
                clock0 = clock::now();
                    CUDA_ERROR( cudaMemcpy( dpData, pData, Nx*Ny*sizeof( dpData[0] ), cudaMemcpyHostToDevice ) );
                clock1 = clock::now();
                seconds = duration_cast<duration<float>>( clock1 - clock0 );
                timeCudaMemcpy.push_back( seconds.count() * 1000 );

                /* shrinkWrap */
                clock0 = clock::now();
                #ifdef USE_FFTW
                    shrinkWrap( pData, Nx, Ny, nShrinkWrapCycles, FLT_MIN ); /* use FLT_MIN to specifiy it should never end or onfly after 32 iterations */
                #endif
                clock1 = clock::now();
                seconds = duration_cast<duration<float>>( clock1 - clock0 );
                timeShrinkWrap.push_back( seconds.count() * 1000 );

                /* cudaShrinkWrap */
                cudaEventRecord( start );
                    cudaShrinkWrap(
                        libs::CudaKernelConfig{
                            12288/256,      /* nBlocks  */
                            256,            /* nThreads */
                            -1,             /* sharedMemBytes (auto) */
                            cudaStream_t(0)
                        },
                        pData, Nx, Ny, nShrinkWrapCycles, FLT_MIN
                    );
                cudaEventRecord( stop );
                cudaEventSynchronize( stop );
                float milliseconds;
                cudaEventElapsedTime( &milliseconds, start, stop );
                timeCudaShrinkWrap.push_back( milliseconds );

                /* cudaFree */
                clock0 = clock::now();
                    CUDA_ERROR( cudaFree( dpData ) );
                clock1 = clock::now();
                seconds = duration_cast<duration<float>>( clock1 - clock0 );
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
                        imresh::io::addTask(
                            /* dummy lambda to omit writing out */
                            []( float *, unsigned int, unsigned int, std::string ){},
                            "",     /* output file name */
                            tmpDataArray + i*Nx*Ny,
                            Nx, Ny, /* image dimensions */
                            nShrinkWrapCycles
                        );
                    }
                    imresh::io::taskQueueDeinit( ); // synchronizes device
                }
                clock1 = clock::now();
                seconds = duration_cast<duration<float>>( clock1 - clock0 );
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


} // namespace algorithms
} // namespace imresh
