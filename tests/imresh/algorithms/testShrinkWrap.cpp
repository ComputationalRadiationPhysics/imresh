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
#include <cuda_runtime_api.h>   // cudaMalloc, cudaMemcpy, cudaEventRecord, ...
#include "algorithms/shrinkWrap.hpp"
#include "algorithms/cuda/cudaShrinkWrap.h"
#include "benchmarkHelper.hpp"  // getLogSpacedSamplingPoints
#include "examples/createTestData/createAtomCluster.hpp"
#include "libs/diffractionIntensity.hpp"
#include "libs/cudacommon.h"


namespace imresh
{
namespace algorithms
{


    /* in order to filter out page time outs or similarly long random wait
     * times, we repeat the measurement nRepetitions times and choose the
     * shortest duration measured */
    unsigned int constexpr nRepetitions = 5;
    unsigned int constexpr nMaxElements = 1024*1024;  // ~8000 x 8000 px
    unsigned int constexpr nShrinkWrapCycles = 4;

    void testShrinkWrap( void )
    {
        using namespace std::chrono;
        using namespace imresh::algorithms;
        using namespace imresh::algorithms::cuda;
        using examples::createTestData::createAtomCluster;
        using imresh::libs::diffractionIntensity;
        using imresh::tests::getLogSpacedSamplingPoints;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        using clock = std::chrono::high_resolution_clock;
        decltype( clock::now() ) clock0, clock1;
        duration<double> seconds;

        std::cout << "\n";
        std::cout << "Timings in milliseconds:\n";
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

            float minTimeCudaMalloc     = FLT_MAX;
            float minTimeCudaMemcpy     = FLT_MAX;
            float minTimeShrinkWrap     = FLT_MAX;
            float minTimeCudaShrinkWrap = FLT_MAX;
            float minTimeCudaFree       = FLT_MAX;
            for ( auto iRepetition = 0u; iRepetition < nRepetitions;
                  ++iRepetition )
            {
                /* cudaMalloc */
                clock0 = clock::now();
                    CUDA_ERROR( cudaMalloc( (void**) &dpData, Nx*Ny*sizeof( dpData[0] ) ) );
                clock1 = clock::now();
                seconds = duration_cast<duration<double>>( clock1 - clock0 );
                minTimeCudaMalloc = fmin( minTimeCudaMalloc, seconds.count() * 1000 );

                /* cudaMemcpy */
                clock0 = clock::now();
                    CUDA_ERROR( cudaMemcpy( dpData, pData, Nx*Ny*sizeof( dpData[0] ), cudaMemcpyHostToDevice ) );
                clock1 = clock::now();
                seconds = duration_cast<duration<double>>( clock1 - clock0 );
                minTimeCudaMemcpy = fmin( minTimeCudaMemcpy, seconds.count() * 1000 );

                /* shrinkWrap */
                clock0 = clock::now();
                    shrinkWrap( pData, Nx, Ny, nShrinkWrapCycles, FLT_MIN ); /* use FLT_MIN to specifiy it should never end or onfly after 32 iterations */
                clock1 = clock::now();
                seconds = duration_cast<duration<double>>( clock1 - clock0 );
                minTimeShrinkWrap = fmin( minTimeShrinkWrap, seconds.count() * 1000 );

                /* cudaShrinkWrap */
                cudaEventRecord( start );
                    cudaShrinkWrap( pData, Nx, Ny, cudaStream_t(0), 12288/256, 256, nShrinkWrapCycles, FLT_MIN );
                cudaEventRecord( stop );
                cudaEventSynchronize( stop );
                float milliseconds;
                cudaEventElapsedTime( &milliseconds, start, stop );
                minTimeCudaShrinkWrap = fmin( minTimeCudaShrinkWrap, milliseconds );

                /* cudaFree */
                clock0 = clock::now();
                    CUDA_ERROR( cudaFree( dpData ) );
                clock1 = clock::now();
                seconds = duration_cast<duration<double>>( clock1 - clock0 );
                minTimeCudaFree = fmin( minTimeCudaFree, seconds.count() * 1000 );
            }
            std::cout << std::setw(10) << minTimeCudaMalloc << " | ";
            std::cout << std::setw(10) << minTimeCudaMemcpy << " | ";
            std::cout << std::setw(10) << minTimeShrinkWrap << " | ";
            std::cout << std::setw(10) << minTimeCudaShrinkWrap << " | ";
            std::cout << std::setw(10) << minTimeCudaFree << " | ";
            std::cout << std::endl;

            delete[] pData;
        }
    }


} // namespace algorithms
} // namespace imresh
