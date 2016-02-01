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
#include <cstdint>              // uint32_t
#include <cstddef>              // NULL


namespace benchmark
{
namespace imresh
{
namespace algorithms
{
namespace cuda
{


    #define CUDA_VECTOR_MAX_DECLARATION( NAME )  \
    template<class T_PREC>                       \
    T_PREC cudaVectorMax##NAME                   \
    (                                            \
        T_PREC const * const rdpData,            \
        unsigned int const rnElements,           \
        cudaStream_t rStream = 0                 \
    );

    CUDA_VECTOR_MAX_DECLARATION( GlobalAtomic2 )
    CUDA_VECTOR_MAX_DECLARATION( GlobalAtomic )
    CUDA_VECTOR_MAX_DECLARATION( SharedMemory )
    CUDA_VECTOR_MAX_DECLARATION( SharedMemoryWarps )


    template<class T_COMPLEX>
    __global__ void cudaKernelCalculateHioErrorBitPacked
    (
        T_COMPLEX const * const __restrict__ rdpgPrime,
        uint32_t  const * const __restrict__ rdpIsMasked,
        unsigned int const rnData,
        float * const __restrict__ rdpTotalError,
        float * const __restrict__ rdpnMaskedPixels
    );

    template<class T_COMPLEX>
    float cudaCalculateHioErrorBitPacked
    (
        T_COMPLEX const * rdpData,
        uint32_t  const * rdpIsMasked,
        unsigned int rnElements,
        bool rInvertMask = false,
        cudaStream_t rStream = 0,
        float * rpTotalError = NULL,
        float * rpnMaskedPixels = NULL
    );


} // namespace cuda
} // namespace algorithms
} // namespace imresh
} // namespace benchmark
