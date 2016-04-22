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


unsigned int constexpr nRepetitions = 20;


int main(void)
{
    using namespace std::chrono;
    using namespace benchmark::imresh::algorithms::cuda;
    using namespace imresh::algorithms;
    using namespace imresh::algorithms::cuda;
    using namespace imresh::libs;
    using clock = std::chrono::high_resolution_clock;

    const unsigned nMaxElements = 64*1024*1024;  // ~4000x4000 pixel
    auto pData = new float[nMaxElements];

    srand(350471643);
    for ( unsigned i = 0; i < nMaxElements; ++i )
        pData[i] = ( (float) rand() / RAND_MAX ) - 0.5f;
    float * dpData;
    mallocCudaArray( &dpData, nMaxElements );
    CUDA_ERROR( cudaMemcpy( dpData, pData, nMaxElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

    /* When commenting one or some of these 6 calls out, the program doesn't
     * seem to deadlock anymore */
    assert( vectorMin( pData, 1 ) == pData[0] );
    assert( vectorMax( pData, 1 ) == pData[0] );
    assert( vectorSum( pData, 1 ) == pData[0] );
    assert( cudaVectorMin( CudaKernelConfig(), dpData, 1 ) == pData[0] );
    assert( cudaVectorMax( CudaKernelConfig(), dpData, 1 ) == pData[0] );
    assert( cudaVectorSum( CudaKernelConfig(), dpData, 1 ) == pData[0] );

    /* do some checks with longer arrays and obvious results */
    float obviousMaximum = 7.37519;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector< unsigned int > values = { 2, 3, 4, 5, 7, 11, 15, 22, 31, 45, 63, 90, 127, 181, 255, 362, 511, 724, 1023, 1448, 2047, 2896, 4095, 5792, 8191, 11585, 16383, 23170, 32767, 46340, 65535, 92681, 131071, 185363, 262143, 370727, 524287, 741455, 1048575, 1482910, 2097151, 2965820, 4194303, 5931641, 8388607, 11863282, 16777214, 23726564, 33554428, 67108864 };

    for ( auto nElements : values )
    {
        std::cout << std::setw(8) << nElements << " : ";
        float milliseconds, minTime;
        decltype( clock::now() ) clock0, clock1;

        int iObviousValuePos = rand() % nElements;

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
        TIME_GPU( cudaVectorMaxGlobalAtomic, obviousMaximum )
        /* When using TIME_CPU insteasd of TIME_GPU the program doesn't
         * seem to hang either ... */
        #define TIME_CPU( FUNC, OBVIOUS_VALUE )                          \
        {                                                                \
            minTime = FLT_MAX;                                           \
            for ( unsigned iRepetition = 0; iRepetition < nRepetitions;  \
                  ++iRepetition )                                        \
            {                                                            \
                clock0 = clock::now();                                   \
                auto cpuMax = FUNC( CudaKernelConfig(), dpData, nElements ); \
                clock1 = clock::now();                                   \
                auto seconds = duration_cast<duration<float>>(           \
                                    clock1 - clock0 );                   \
                minTime = std::fmin( minTime, seconds.count() * 1000 );  \
                assert( cpuMax == OBVIOUS_VALUE );                       \
            }                                                            \
            std::cout << std::setw(8) << minTime << " |" << std::flush;  \
        }
        //TIME_CPU( cudaVectorMaxGlobalAtomic, obviousMaximum )
        std::cout << std::endl;
    }

    CUDA_ERROR( cudaFree( dpData ) );
    delete[] pData;

    return 0;
}
