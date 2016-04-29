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


unsigned int constexpr nRepetitions = 20;


int main(void)
{
    using namespace std::chrono;
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
    assert( vectorMax( pData, 1 ) == pData[0] );
    assert( cudaVectorMax( CudaKernelConfig(), dpData, 1 ) == pData[0] );

    /* do some checks with longer arrays and obvious results */
    float obviousMaximum = 7.37519;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector< unsigned int > values = { 2, 3, 4, 5, 7, 11, 15, 22, 31, 45, 63, 90, 127, 181, 255, 362, 511, 724, 1023, 1448, 2047, 2896, 4095, 5792, 8191, 11585, 16383, 23170, 32767, 46340, 65535, 92681, 131071, 185363, 262143, 370727, 524287, 741455, 1048575, 1482910, 2097151, 2965820, 4194303, 5931641, 8388607, 11863282, 16777214, 23726564 }; //, 33554428, 67108864 };

    std::cout
    << "         : local +  |local  + | ibid.   |         |         | #pragma |\n"
    << "         : glob.atom|shared + |(old warp|local +  |         | omp     |\n"
    << "nElements: ptr aritm|glob.atom| ver.)   |glob.atom|glob.atom| reduce  |\n";

    for ( auto nElements : values )
    {
        std::cout << std::setw(8) << nElements << " : ";
        float milliseconds, minTime;
        decltype( clock::now() ) clock0, clock1;

        int iObviousValuePos = rand() % nElements;

        /* Maximum */
        pData[iObviousValuePos] = obviousMaximum;
        CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

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
                if ( cpuMax != OBVIOUS_VALUE )                           \
                    assert( cpuMax == OBVIOUS_VALUE );                   \
            }                                                            \
            std::cout << std::setw(8) << minTime << " |" << std::flush;  \
        }
        TIME_CPU( cudaVectorMax, obviousMaximum )
        TIME_CPU( cudaVectorMaxSharedMemory, obviousMaximum )
        TIME_CPU( cudaVectorMaxSharedMemoryWarps, obviousMaximum )
        TIME_CPU( cudaVectorMaxGlobalAtomic, obviousMaximum )
        if ( nElements < 1e6 )
            TIME_CPU( cudaVectorMaxGlobalAtomic2, obviousMaximum )
        else
            std::cout << std::setw(8) << -1 << " |" << std::flush;
        #undef TIME_CPU

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
                if( cpuMax == OBVIOUS_VALUE )                            \
                    assert( cpuMax == OBVIOUS_VALUE );                   \
            }                                                            \
            std::cout << std::setw(8) << minTime << " |" << std::flush;  \
        }

        TIME_CPU( vectorMax, obviousMaximum )

        std::cout << std::endl;
    }

    CUDA_ERROR( cudaFree( dpData ) );
    delete[] pData;

    return 0;
}
