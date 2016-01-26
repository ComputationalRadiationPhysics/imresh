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


namespace imresh
{
namespace algorithms
{
namespace cuda
{


    template< class T_COMPLEX, class T_PREC >
    __global__ void cudaKernelCopyToRealPart
    (
        T_COMPLEX * const rTargetComplexArray,
        T_PREC    * const rSourceRealArray,
        unsigned    const rnElements
    );

    template< class T_PREC, class T_COMPLEX >
    __global__ void cudaKernelCopyFromRealPart
    (
        T_PREC    * const rTargetComplexArray,
        T_COMPLEX * const rSourceRealArray,
        unsigned    const rnElements
    );

    template< class T_PREC, class T_COMPLEX >
    __global__ void cudaKernelComplexNormElementwise
    (
        T_PREC * const rdpDataTarget,
        const T_COMPLEX * const rdpDataSource,
        const unsigned rnElements
    );

    template< class T_COMPLEX, class T_PREC >
    __global__ void cudaKernelApplyComplexModulus
    (
        T_COMPLEX * const rdpDataTarget,
        const T_COMPLEX * const rdpDataSource,
        const T_PREC * const rdpComplexModulus,
        const unsigned rnElements
    );

    template< class T_PREC >
    __global__ void cudaKernelCutOff
    (
        T_PREC * const rData,
        unsigned const rnElements,
        const T_PREC rThreshold,
        const T_PREC rLowerValue,
        const T_PREC rUpperValue
    );

    /* kernel call wrappers in order for this to be usable from source files
     * not compiled with nvcc */

    template< class T_PREC, class T_COMPLEX >
    void cudaComplexNormElementwise
    (
        T_PREC * const rdpDataTarget,
        const T_COMPLEX * const rdpDataSource,
        const unsigned rnElements,
        const cudaStream_t rStream,
        const bool rAsync
    );


} // namespace cuda
} // namespace algorithms
} // namespace imresh
