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


#include <iostream>
#include <cassert>
#include <cstdlib>   // srand, rand
#include <cuda_runtime.h>
#include "algorithms/cuda/cudaVectorReduce.hpp"
#include "libs/cudacommon.hpp"

#define USE_PINNED_MEMORY 0
#define MALLOC_BEFORE_MEMCPY 0 // if 0 program takes 1.4s instead of 0.9s O_O!? in nvvp the difference looks even worse for some reason (1.25s vs 0.125s), only difference visible if not using pinned memory!
/* Note that in this case (~270MB) the overall time needed for the execution
 * stays the same, because cudaFreeHost is thousands of times slower than
 * delete[] -.- ... Would only be usefull for more than one transfer I guess
 * which we have if we do our own memory managment in order to avoid cudaMalloc
 *
 * All this is getting ridiculous -.-
 */


int main( void )
{
    using namespace imresh::libs; // mallocCudaArray, mallocPinnedArray

    const unsigned nElements = 64*1024*1024;  // ~4000x4000 pixel

    /* using cudaMallocHost (pinned memory) instead of paged memory, increased
     * cudaMemcpy bandwidth from 4.7GB/s to 6.7GB/s! */
    float * pData, * dpData;
    #if MALLOC_BEFORE_MEMCPY == 0
        mallocCudaArray( &dpData, nElements );
    #endif
    #if USE_PINNED_MEMORY != 0
        mallocPinnedArray( &pData, nElements );
    #else
        pData = new float[nElements];
    #endif

    srand(350471643);
    for ( unsigned i = 0; i < nElements; ++i )
        pData[i] = ( (float) rand() / RAND_MAX ) - 0.5f;
    const int iObviousValuePos = rand() % nElements;
    const float obviousMaximum = 7.37519;
    pData[iObviousValuePos] = obviousMaximum;

    #if MALLOC_BEFORE_MEMCPY != 0
        mallocCudaArray( &dpData, nElements );
    #endif
    CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

    auto cudaMax = imresh::algorithms::cuda::cudaVectorMax( dpData, nElements );
    if( cudaMax == obviousMaximum )
        std::cout << "Result seems correct.\n";
    else
        std::cout << "Wrong result!\n";

    CUDA_ERROR( cudaFree( dpData ) );
    #if USE_PINNED_MEMORY == 1
        CUDA_ERROR( cudaFreeHost( pData ) );
    #else
        delete[] pData;
    #endif
}
