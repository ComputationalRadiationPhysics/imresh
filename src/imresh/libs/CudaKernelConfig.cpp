/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Maximilian Knespel
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


#include "CudaKernelConfig.hpp"

#include <cuda_to_cupla.hpp>    // cudaStream_t


namespace imresh
{
namespace libs
{


    CudaKernelConfig::CudaKernelConfig
    (
        dim3         rnBlocks         ,
        dim3         rnThreads        ,
        int          rnBytesSharedMem ,
        cudaStream_t riCudaStream
    )
    :
        nBlocks        ( rnBlocks         ),
        nThreads       ( rnThreads        ),
        nBytesSharedMem( rnBytesSharedMem ),
        iCudaStream    ( riCudaStream     )
    {
        check();
    }

    int CudaKernelConfig::check( void )
    {
        int changed = 0;
        int nMaxBlocks, nMaxThreads;

        #if defined( ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED    ) || \
            defined( ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED    ) || \
            defined( ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED ) || \
            defined( ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED     )
            nMaxBlocks  = 1; // number of cores?
            nMaxThreads = 4;
        #endif
        #ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
            nMaxThreads = 128;
            nMaxBlocks  = 192;    // MaxConcurrentThreads / nMaxThreads, e.g. on GTX760 it would be 96
        #endif

        if ( nBlocks.x <= 0 )
        {
            changed += 1;
            nBlocks.x = nMaxBlocks;
        }
        if ( nBlocks.y <= 0 )
        {
            changed += 1;
            nBlocks.y = 1;
        }
        if ( nBlocks.z <= 0 )
        {
            changed += 1;
            nBlocks.z = 1;
        }

        if ( nThreads.x <= 0 )
        {
            changed += 1;
            nThreads.x = nMaxThreads;
        }
        if ( nThreads.y <= 0 )
        {
            changed += 1;
            nThreads.y = 1;
        }
        if ( nThreads.z <= 0 )
        {
            changed += 1;
            nThreads.z = 1;
        }

        assert( nBytesSharedMem >= 0 );

        return changed;
    }


} // namespace libs
} // namespace imresh
