
#include <cstdlib>   // srand, rand
#include <cuda_runtime.h>
#include "algorithms/cuda/cudaGaussian.h"
#include "libs/cudacommon.h"


int main()
{
    using namespace imresh::algorithms::cuda;

    const unsigned nMaxElements = 4*1024*1024;

    /* fill test data with random numbers from [-0.5,0.5] */
    float * pData, *dpData;
    CUDA_ERROR( cudaMallocHost( (void**) &pData, nMaxElements*sizeof(pData[0]) ) );
    CUDA_ERROR( cudaMalloc( (void**)&dpData, nMaxElements*sizeof(dpData[0]) ) );

    srand(350471643);
    for ( unsigned i = 0; i < nMaxElements; ++i )
        pData[i] = (float) rand() / RAND_MAX - 0.5;
    CUDA_ERROR( cudaMemcpy( dpData, pData, nMaxElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

    cudaGaussianBlurHorizontal( dpData, 2048, 2048, 3.0 );

    CUDA_ERROR( cudaFree( dpData ) );
    CUDA_ERROR( cudaFreeHost( pData ) );
}
