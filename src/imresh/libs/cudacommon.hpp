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

#include <cstdlib>      // malloc, free
#include <cuda.h>       // cudaStream_t
#include <cuda_runtime_api.h>
#include <cassert>


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

    template< class T >
    struct GpuArray
    {
        T * host, * gpu;
        unsigned long long int const nBytes;
        cudaStream_t mStream;

        inline GpuArray
        (
            unsigned long long int const nElements = 1,
            cudaStream_t rStream = 0
        )
        : nBytes( nElements * sizeof(T) ),
          mStream( rStream )
        {
            host = (T*) malloc( nBytes );
            CUDA_ERROR( cudaMalloc( (void**) &gpu, nBytes ) );
            assert( host != NULL );
            assert( gpu  != NULL );
        }
        inline ~GpuArray()
        {
            CUDA_ERROR( cudaFree( gpu ) );
            free( host );
        }
        inline void down( void )
        {
            CUDA_ERROR( cudaMemcpyAsync( (void*) host, (void*) gpu, nBytes, cudaMemcpyDeviceToHost ) );
            CUDA_ERROR( cudaPeekAtLastError() );
        }
        inline void up( void )
        {
            CUDA_ERROR( cudaMemcpyAsync( (void*) gpu, (void*) host, nBytes, cudaMemcpyHostToDevice ) );
            CUDA_ERROR( cudaPeekAtLastError() );
        }
    };


} // namespace libs
} // namespace imresh
