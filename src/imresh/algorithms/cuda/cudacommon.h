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


#pragma once

#include <chrono>
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>


namespace imresh
{
namespace algorithms
{
namespace cuda
{


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
    #define CUDA_ERROR(X) imresh::algorithms::cuda::checkCudaError(X,__FILE__,__LINE__);

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


} // namespace cuda
} // namespace algorithms
} // namespace imresh
