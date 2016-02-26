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

#include <cuda_to_cupla.hpp>


namespace imresh
{
namespace algorithms
{

    int dummy(void);

    template< class T_COMPLEX, class T_PREC >
    struct cudaKernelApplyHioDomainConstraints
    {
        template< typename T_ACC >
        ALPAKA_FN_ACC
        void operator()
        (
            T_ACC const & acc,
            T_COMPLEX       * const rdpgPrevious,
            T_COMPLEX const * const rdpgPrime,
            T_PREC    const * const rdpIsMasked,
            unsigned int const rnElements,
            T_PREC const rHioBeta
        ) const;
    };

#if false
    template< class T_COMPLEX, class T_PREC >
    __global__ void cudaKernelApplyHioDomainConstraints
    (
        T_COMPLEX       * const rdpgPrevious,
        T_COMPLEX const * const rdpgPrime,
        T_PREC    const * const rdpIsMasked,
        unsigned int const rnElements,
        T_PREC rHioBeta
    );

    template< class T_COMPLEX, class T_PREC >
    __global__ void cudaKernelCopyToRealPart
    (
        T_COMPLEX * const rTargetComplexArray,
        T_PREC    * const rSourceRealArray,
        unsigned int const rnElements
    );

    template< class T_PREC, class T_COMPLEX >
    __global__ void cudaKernelCopyFromRealPart
    (
        T_PREC    * const rTargetComplexArray,
        T_COMPLEX * const rSourceRealArray,
        unsigned int const rnElements
    );

    template< class T_PREC, class T_COMPLEX >
    __global__ void cudaKernelComplexNormElementwise
    (
        T_PREC          * const rdpDataTarget,
        T_COMPLEX const * const rdpDataSource,
        unsigned int const rnElements
    );

    template< class T_COMPLEX, class T_PREC >
    __global__ void cudaKernelApplyComplexModulus
    (
        T_COMPLEX       * const rdpDataTarget,
        T_COMPLEX const * const rdpDataSource,
        T_PREC    const * const rdpComplexModulus,
        unsigned int const rnElements
    );

    template< class T_PREC >
    __global__ void cudaKernelCutOff
    (
        T_PREC * const rData,
        unsigned int const rnElements,
        T_PREC const rThreshold,
        T_PREC const rLowerValue,
        T_PREC const rUpperValue
    );

    /* kernel call wrappers in order for this to be usable from source files
     * not compiled with nvcc */

    template< class T_PREC, class T_COMPLEX >
    void cudaComplexNormElementwise
    (
        T_PREC          * const rdpDataTarget,
        T_COMPLEX const * const rdpDataSource,
        unsigned int const rnElements,
        cudaStream_t const rStream = cudaStream_t(0),
        bool const rAsync = true
    );
#endif

} // namespace algorithms
} // namespace imresh
