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
#include <cuda_to_cupla.hpp>
#include "algorithms/vectorReduce.hpp"
#include "algorithms/cuda/cudaVectorReduce.hpp"
#include "benchmark/imresh/algorithms/cuda/cudaVectorReduce.hpp"
#include "libs/cudacommon.hpp"
#include "benchmarkHelper.hpp"


unsigned int constexpr nRepetitions = 20;


void testVectorReduce( void )
{
    using namespace std::chrono;
    using namespace benchmark::imresh::algorithms::cuda;
    using namespace imresh::algorithms;
    using namespace imresh::algorithms::cuda;
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
    assert( cudaVectorMin( CudaKernelConfig(), dpData, 1 ) == pData[0] );
    assert( cudaVectorMax( CudaKernelConfig(), dpData, 1 ) == pData[0] );
    assert( cudaVectorSum( CudaKernelConfig(), dpData, 1 ) == pData[0] );

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

    std::cout << "# Timings are in milliseconds, but note that measurements are repeated " << nRepetitions << " times, meaning they take that much longer than the value displayed\n";
    std::cout <<
        "# vector : cudaVec-| cudaVec- | cudaVec-| cudaVec-| cudaVec-| CPU vec-|\n"
        "# length : torMax  | torMax   | torMax  | torMax  | torMax  | torMax  |\n"
        "#        : global  | global   | shared  | shared  | __shfl_ |         |\n"
        "#        : atomic  | atomic2  | memory  | + warp  |   down  |         |\n";
        /*
               90 :  1.03741 | 1.03846 | 1.03817 | 1.03809 | 1.04916 |0.006181 |
        */

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
                auto cudaReduced = FUNC( CudaKernelConfig(), dpData,     \
                                         nElements );                    \
                cudaEventRecord( stop );                                 \
                cudaEventSynchronize( stop );                            \
                cudaEventElapsedTime( &milliseconds, start, stop );      \
                minTime = std::fmin( minTime, milliseconds );            \
                assert( cudaReduced == OBVIOUS_VALUE );                  \
            }                                                            \
            std::cout << std::setw(8) << minTime << " |" << std::flush;  \
        }

        //TIME_GPU( cudaVectorMaxGlobalAtomic2    , obviousMaximum )
        TIME_GPU( cudaVectorMaxGlobalAtomic     , obviousMaximum )
        //TIME_GPU( cudaVectorMaxSharedMemory     , obviousMaximum )
        //TIME_GPU( cudaVectorMaxSharedMemoryWarps, obviousMaximum )
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
                minTime = std::fmin( minTime, seconds.count() * 1000 );  \
                assert( cpuMax == OBVIOUS_VALUE );                       \
            }                                                            \
            std::cout << std::setw(8) << minTime << " |" << std::flush;  \
        }
        TIME_CPU( vectorMax, obviousMaximum )

        /* Minimum *//*
        pData[iObviousValuePos] = obviousMinimum;
        CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

        TIME_GPU( cudaVectorMin, obviousMinimum )*/
        //TIME_CPU( vectorMin, obviousMinimum )

        /* set obvious value back to random value */
        pData[iObviousValuePos] = (float) rand() / RAND_MAX;
        std::cout << "\n";

        #undef TIME_GPU
        #undef TIME_CPU
    }

    CUDA_ERROR( cudaFree( dpData ) );
    delete[] pData;
}

int main(void)
{
    testVectorReduce();
    return 0;
}
