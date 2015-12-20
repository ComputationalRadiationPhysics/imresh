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

void checkCudaError(const cudaError_t rValue, const char * file, int line )
{
    if ( (rValue) != cudaSuccess )
    std::cout << "CUDA error in " << file
              << " line:" << line << " : "
              << cudaGetErrorString(rValue) << "\n";
}
#define CUDA_ERROR(X) checkCudaError(X,__FILE__,__LINE__);
/* not not impossible, but also not beautiful implementing this    *
 * as a pure macro, because we need to temporarily save the return *
 * value, else the function would be evaluated twice, once for     *
 * checking, and the second time for printing!                     */




/**
 * @param[out] rpDeviceProperties - Array of cudaDeviceProp of length rnDevices
 *             the user will need to free (C-style) this data on program exit!
 * @param[out] rnDevices - will hold number of cuda devices
 **/
void getCudaDeviceProperties( cudaDeviceProp** rpDeviceProperties, int * rnDevices, bool rPrintInfo = false )
{
    printf("Getting Device Informations. As this is the first command, "
           "it can take ca.30s, because the GPU must be initialized.\n");
    fflush(stdout);

    int fallbackNDevices;
    if (rnDevices == NULL)
        rnDevices = &fallbackNDevices;
    cudaGetDeviceCount(rnDevices);

    cudaDeviceProp * fallbackPropArray;
    if ( rpDeviceProperties == NULL )
        rpDeviceProperties = &fallbackPropArray;
    *rpDeviceProperties = (cudaDeviceProp*) malloc( (*rnDevices) * sizeof(cudaDeviceProp) );

    for (int iDevice = 0; iDevice < (*rnDevices); iDevice++)
    {
        cudaDeviceProp * prop = &((*rpDeviceProperties)[iDevice]);
        cudaGetDeviceProperties( prop, iDevice);

		if (not rPrintInfo)
			continue;

        if (iDevice == 0 && prop->major == 9999 && prop->minor == 9999)
            printf("There is no device supporting CUDA.\n");

		const char cms[5][20] =
			{ "Default", "Exclusive", "Prohibited", "ExclusiveProcess", "Unknown" };
		const char * computeModeString;
		switch(prop->computeMode) {
			case cudaComputeModeDefault          : computeModeString = cms[0];
			case cudaComputeModeExclusive        : computeModeString = cms[1];
			case cudaComputeModeProhibited       : computeModeString = cms[2];
			case cudaComputeModeExclusiveProcess : computeModeString = cms[3];
			default                              : computeModeString = cms[4];
		}

        printf("\n================== Device Number %i ==================\n",iDevice);
        printf("| Device name              : %s\n"        ,prop->name);
        printf("| Computability            : %i.%i\n"     ,prop->major,
                                                           prop->minor);
        printf("| PCI Bus ID               : %i\n"        ,prop->pciBusID);
        printf("| PCI Device ID            : %i\n"        ,prop->pciDeviceID);
        printf("| PCI Domain ID            : %i\n"        ,prop->pciDomainID);
		printf("|------------------- Architecture -------------------\n");
        printf("| Number of SMX            : %i\n"        ,prop->multiProcessorCount);
        printf("| Max Threads per SMX      : %i\n"        ,prop->maxThreadsPerMultiProcessor);
        printf("| Max Threads per Block    : %i\n"        ,prop->maxThreadsPerBlock);
        printf("| Warp Size                : %i\n"        ,prop->warpSize);
        printf("| Clock Rate               : %f GHz\n"    ,prop->clockRate/1.0e6f);
        printf("| Max Block Size           : (%i,%i,%i)\n",prop->maxThreadsDim[0],
                                                           prop->maxThreadsDim[1],
                                                           prop->maxThreadsDim[2]);
        printf("| Max Grid Size            : (%i,%i,%i)\n",prop->maxGridSize[0],
                                                           prop->maxGridSize[1],
                                                           prop->maxGridSize[2]);
		printf("|  => Max conc. Threads    : %i\n"        ,prop->multiProcessorCount *
		                                                   prop->maxThreadsPerMultiProcessor);
		printf("|  => Warps per SMX        : %i\n"        ,prop->maxThreadsPerMultiProcessor /
		                                                   prop->warpSize);
		printf("|---------------------- Memory ----------------------\n");
        printf("| Total Global Memory      : %lu Bytes\n" ,prop->totalGlobalMem);
        printf("| Total Constant Memory    : %lu Bytes\n" ,prop->totalConstMem);
        printf("| Shared Memory per Block  : %lu Bytes\n" ,prop->sharedMemPerBlock);
        printf("| L2 Cache Size            : %u Bytes\n"  ,prop->l2CacheSize);
        printf("| Registers per Block      : %i\n"        ,prop->regsPerBlock);
        printf("| Memory Bus Width         : %i Bits\n"   ,prop->memoryBusWidth);
        printf("| Memory Clock Rate        : %f GHz\n"    ,prop->memoryClockRate/1.0e6f);
        printf("| Memory Pitch             : %lu\n"       ,prop->memPitch);
        printf("| Unified Addressing       : %i\n"        ,prop->unifiedAddressing);
		printf("|--------------------- Graphics ---------------------\n");
		printf("| Compute mode             : %s\n"        ,      computeModeString);
		printf("|---------------------- Other -----------------------\n");
        printf("| Can map Host Memory      : %s\n"        ,prop->canMapHostMemory  ? "true" : "false");
        printf("| Can run Kernels conc.    : %s\n"        ,prop->concurrentKernels ? "true" : "false");
		printf("| Number of Asyn. Engines  : %i\n"        ,prop->asyncEngineCount);
        printf("| Can Copy and Kernel conc.: %s\n"        ,prop->deviceOverlap     ? "true" : "false");
        printf("| ECC Enabled              : %s\n"        ,prop->ECCEnabled        ? "true" : "false");
        printf("| Device is Integrated     : %s\n"        ,prop->integrated        ? "true" : "false");
        printf("| Kernel Timeout Enabled   : %s\n"        ,prop->kernelExecTimeoutEnabled ? "true" : "false");
        printf("| Uses TESLA Driver        : %s\n"        ,prop->tccDriver         ? "true" : "false");
        printf("=====================================================\n");
    }
}

/**
 * atomicAdd for double is not natively implemented, because it's not
 * supported by (all) the hardware, therefore resulting in a time penalty.
 * http://stackoverflow.com/questions/12626096/why-has-atomicadd-not-been-implemented-for-doubles
 */
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
              __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

