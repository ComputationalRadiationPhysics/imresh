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

#include <cuda_runtime_api.h>    // cudaStream_t


namespace benchmark
{
namespace imresh
{
namespace algorithms
{
namespace cuda
{


    /**
     * Should only be used for benchmarking purposes.
     *
     * Makes use of constant memory to store the kernel.
     * @see cudaGaussianBlurHorizontal
     **/
    template<class T_PREC>
    void cudaGaussianBlurHorizontalConstantWeights
    (
        T_PREC * rdpData,
        unsigned rnDataX,
        unsigned rnDataY,
        double rSigma,
        cudaStream_t rStream = 0,
        bool rAsync = false
    );


} // namespace cuda
} // namespace algorithms
} // namespace imresh
} // namespace benchmark