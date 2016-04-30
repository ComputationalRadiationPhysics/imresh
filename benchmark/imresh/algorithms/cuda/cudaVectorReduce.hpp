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

#include <cuda_to_cupla.hpp>    // cudaStream_t
#include <cstdint>              // uint32_t
#include <cstddef>              // NULL
#include "libs/CudaKernelConfig.hpp"


namespace benchmark
{
namespace imresh
{
namespace algorithms
{
namespace cuda
{

    using ::imresh::libs::CudaKernelConfig;

    #define CUDA_VECTOR_MAX_DECLARATION( NAME )  \
    template<class T_PREC>                       \
    T_PREC cudaVectorMax##NAME                   \
    (                                            \
        CudaKernelConfig rKernelConfig,          \
        T_PREC const * const rdpData,            \
        unsigned int const rnElements            \
    );
    CUDA_VECTOR_MAX_DECLARATION( GlobalAtomic2 )
    CUDA_VECTOR_MAX_DECLARATION( GlobalAtomic )
    CUDA_VECTOR_MAX_DECLARATION( Pointer )
    CUDA_VECTOR_MAX_DECLARATION( SharedMemory )
    CUDA_VECTOR_MAX_DECLARATION( SharedMemoryWarps )
    #undef CUDA_VECTOR_MAX_DECLARATION


    template<class T_COMPLEX>
    float cudaCalculateHioErrorBitPacked
    (
        CudaKernelConfig rKernelConfig,
        T_COMPLEX const * rdpData,
        uint32_t  const * rdpIsMasked,
        unsigned int rnElements,
        bool rInvertMask = false,
        float * rpTotalError = NULL,
        float * rpnMaskedPixels = NULL
    );


} // namespace cuda
} // namespace algorithms
} // namespace imresh
} // namespace benchmark
