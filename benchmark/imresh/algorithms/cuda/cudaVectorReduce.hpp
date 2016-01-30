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

#include <cuda_runtime_api.h>   // cudaStream_t


namespace benchmark
{
namespace imresh
{
namespace algorithms
{
namespace cuda
{


    /**
     * @see kernelVectorReduceWarps but uses only shared memory to reduce
     * per block
     **/
    template<class T_PREC, class T_FUNC>
    __global__ void kernelVectorReduceShared
    (
        T_PREC const * rdpData,
        unsigned int rnData,
        T_PREC * rdpResult,
        T_FUNC f,
        T_PREC rInitValue
    );

    /**
     * @see kernelVectorReduceWarps but uses also shared memory to reduce
     * per block
     **/
    template<class T_PREC, class T_FUNC>
    __global__ void kernelVectorReduceSharedWarps
    (
        T_PREC const * rdpData,
        unsigned int rnData,
        T_PREC * rdpResult,
        T_FUNC f,
        T_PREC rInitValue
    );

    template<class T_PREC, class T_FUNC>
    T_PREC cudaReduceSharedMemory
    (
        T_PREC const * const rdpData,
        unsigned int const rnElements,
        T_FUNC f,
        T_PREC const rInitValue,
        cudaStream_t rStream = 0
    );

    template<class T_PREC, class T_FUNC>
    T_PREC cudaReduceSharedMemoryWarps
    (
        T_PREC const * const rdpData,
        unsigned int const rnElements,
        T_FUNC f,
        T_PREC const rInitValue,
        cudaStream_t rStream = 0
    );

    template<class T_COMPLEX, class T_MASK>
    __global__ void cudaKernelCalculateHioError
    (
        T_COMPLEX const * rdpgPrime,
        T_MASK    const * rdpIsMasked,
        unsigned int rnData,
        bool rInvertMask,
        float * rdpTotalError,
        float * rdpnMaskedPixels
    );

    template<class T_PREC>
    T_PREC cudaVectorMaxSharedMemory
    (
        T_PREC const * rdpData,
        unsigned int rnElements,
        cudaStream_t rStream = 0
    );

    template<class T_PREC>
    T_PREC cudaVectorMaxSharedMemoryWarps
    (
        T_PREC const * rdpData,
        unsigned int rnElements,
        cudaStream_t rStream = 0
    );

    template<class T_COMPLEX>
    __global__ void cudaKernelCalculateHioErrorBitPacked
    (
        T_COMPLEX const * rdpgPrime,
        int       const * rdpIsMasked,
        unsigned int rnData,
        float * rdpTotalError,
        float * rdpnMaskedPixels
    );


    float _instantiateAllTemplatesCudaVectorReduceBenchmark(void);


} // namespace cuda
} // namespace algorithms
} // namespace imresh
} // namespace benchmark
