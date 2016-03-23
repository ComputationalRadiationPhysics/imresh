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


#pragma once


#include <cuda_to_cupla.hpp>    // cudaStream_t


namespace imresh
{
namespace libs
{


    struct CudaKernelConfig
    {
        dim3          nBlocks;
        dim3          nThreads;
        int           nBytesSharedMem;
        cudaStream_t  iCudaStream;

        /**
         *
         * Note thate CudaKernelConfig( 1,1,0, cudaStream_t(0) ) is still
         * possible, because dim3( int ) is declared.
         */
        CudaKernelConfig
        (
            dim3         rnBlocks         = dim3{ 0, 0, 0 },
            dim3         rnThreads        = dim3{ 0, 0, 0 },
            int          rnBytesSharedMem = -1                ,
            cudaStream_t riCudaStream     = cudaStream_t(0)
        );

        /**
         * Checks configuration and autoadjusts default or faulty parameters
         *
         * @return 0: everything was OK, 1: some parameters had to be changed
         */
        int check( void );
    };


} // namespace libs
} // namespace imresh
