#pragma once

#include <chrono>
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>


//#define sleep(DELTAT) std::this_thread::sleep_for(std::chrono::milliseconds(DELTAT))


__device__ inline uint64_t getLinearThreadId(void)
{
    return threadIdx.x +
           threadIdx.y * blockDim.x +
           threadIdx.z * blockDim.x * blockDim.y;
}

__device__ inline uint64_t getLinearId(void)
{
    uint64_t linId = threadIdx.x;
    uint64_t maxSize = blockDim.x;
    linId += maxSize*threadIdx.y;
    maxSize *= blockDim.y;
    linId += maxSize*threadIdx.z;
    maxSize *= blockDim.z;
    linId += maxSize*blockIdx.x;
    maxSize *= gridDim.x;
    linId += maxSize*blockIdx.y;
    maxSize *= gridDim.y;
    linId += maxSize*blockIdx.z;
    // maxSize *= gridDim.z;
    return linId;
}

__device__ inline uint64_t getDim3Product( const dim3 & rVec )
{ return (uint64_t)rVec.x * rVec.y * rVec.z; }

__device__ inline uint64_t getBlockSize( void )
{ return getDim3Product(blockDim); }

__device__ inline uint64_t getGridSize( void )
{ return getDim3Product(gridDim); }

void checkCudaError(const cudaError_t rValue, const char * file, int line );
#define CUDA_ERROR(X) checkCudaError(X,__FILE__,__LINE__);

/**
 * @param[out] rpDeviceProperties - Array of cudaDeviceProp of length rnDevices
 *             the user will need to free (C-style) this data on program exit!
 * @param[out] rnDevices - will hold number of cuda devices
 **/
void getCudaDeviceProperties
( cudaDeviceProp** rpDeviceProperties, int * rnDevices, bool rPrintInfo = false );

/**
 * atomicAdd for double is not natively implemented, because it's not
 * supported by (all) the hardware, therefore resulting in a time penalty.
 * http://stackoverflow.com/questions/12626096/why-has-atomicadd-not-been-implemented-for-doubles
 */
__device__ double atomicAdd(double* address, double val);
