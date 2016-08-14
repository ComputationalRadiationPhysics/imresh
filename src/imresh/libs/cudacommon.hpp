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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdlib>


#define CUDA_ERROR(X) ::imresh::libs::checkCudaError(X,__FILE__,__LINE__);

namespace imresh
{
namespace libs
{


    void checkCudaError(const cudaError_t rValue, const char * file, int line );

    template< typename T >
    inline void mallocCudaArray( T ** const rPtr, unsigned int const rnElements )
    {
        assert( rnElements > 0 );
        CUDA_ERROR( cudaMalloc( (void**) rPtr, sizeof(T) * rnElements ) );
        assert( rPtr != NULL );
    }

    template< typename T >
    inline void mallocPinnedArray( T ** const rPtr, unsigned int const rnElements )
    {
        assert( rnElements > 0 );
        CUDA_ERROR( cudaMallocHost( (void**) rPtr, sizeof(T) * rnElements ) );
        assert( rPtr != NULL );
    }


    /**
     * Basically saves a gpu/cpu pointer pair plus their size and therefore
     * makes it a tad shorter to communicate with a GPU device
     */
    template< typename T >
    struct HostDeviceMemory
    {
        T * cpu, * gpu;
        unsigned int nBytes;

        inline HostDeviceMemory( int const nElements )
        {
            nBytes = sizeof(T) * nElements;
            cpu = (T*) malloc( nBytes );
            CUDA_ERROR( cudaMalloc( &gpu, nBytes ) );
        }

        inline ~HostDeviceMemory()
        {
            free( cpu );
            cpu = NULL;
            CUDA_ERROR( cudaFree( gpu ) );
            gpu = NULL;
        }

        inline void toGpu( void )
        {
            assert( cpu != NULL );
            assert( gpu != NULL );
            CUDA_ERROR( cudaMemcpy( cpu, gpu, nBytes, cudaMemcpyHostToDevice ) );
        }

        inline void toCpu( void )
        {
            assert( cpu != NULL );
            assert( gpu != NULL );
            CUDA_ERROR( cudaMemcpy( gpu, cpu, nBytes, cudaMemcpyDeviceToHost ) );
        }
    };


} // namespace libs
} // namespace imresh
