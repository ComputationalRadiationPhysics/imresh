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


#include "cudaVectorElementwise.tpp"


namespace imresh
{
namespace algorithms
{
namespace cuda
{


    /* explicit instantiations */

    template
    __global__ void cudaKernelApplyHioDomainConstraints<cufftComplex, float>
    (
        cufftComplex       * const rdpgPrevious,
        cufftComplex const * const rdpgPrime,
        float const * const rdpIsMasked,
        unsigned int const rnElements,
        float const rHioBeta
    );

    template
    __global__ void cudaKernelCopyToRealPart<cufftComplex,float>
    (
        cufftComplex * const rTargetComplexArray,
        float * const rSourceRealArray,
        unsigned int const rnElements
    );


    template
    __global__ void cudaKernelCopyFromRealPart<float,cufftComplex>
    (
        float * const rTargetComplexArray,
        cufftComplex * const rSourceRealArray,
        unsigned int const rnElements
    );


    #define INSTANTIATE_cudaKernelComplexNormElementwise( T_PREC, T_COMPLEX ) \
    template                                                                  \
    __global__ void cudaKernelComplexNormElementwise<T_PREC,T_COMPLEX>        \
    (                                                                         \
        T_PREC * const rdpDataTarget,                                         \
        T_COMPLEX const * const rdpDataSource,                                \
        unsigned int const rnElements                                         \
    );
    INSTANTIATE_cudaKernelComplexNormElementwise( float, cufftComplex )

    #define INSTANTIATE_cudaComplexNormElementwise( T_PREC, T_COMPLEX ) \
    template                                                            \
    void cudaComplexNormElementwise<T_PREC, T_COMPLEX>                  \
    (                                                                   \
        T_PREC * const rdpDataTarget,                                   \
        T_COMPLEX const * const rdpDataSource,                          \
        unsigned int const rnElements,                                  \
        cudaStream_t const rStream,                                     \
        bool const rAsync                                               \
    );
    INSTANTIATE_cudaComplexNormElementwise( float, cufftComplex )
    INSTANTIATE_cudaComplexNormElementwise( cufftComplex, cufftComplex )

    #define INSTANTIATE_cudaKernelApplyComplexModulus( T_COMPLEX, T_PREC )  \
    template                                                                \
    __global__ void cudaKernelApplyComplexModulus<T_COMPLEX,T_PREC>         \
    (                                                                       \
        T_COMPLEX * const rdpDataTarget,                                    \
        T_COMPLEX const * const rdpDataSource,                              \
        T_PREC const * const rdpComplexModulus,                             \
        unsigned int const rnElements                                       \
    );
    INSTANTIATE_cudaKernelApplyComplexModulus( cufftComplex, float )

    #define INSTANTIATE_cudaKernelCutOff( T_PREC )  \
    template                                        \
    __global__ void cudaKernelCutOff<T_PREC>        \
    (                                               \
        T_PREC * const rData,                       \
        unsigned int const rnElements,              \
        T_PREC const rThreshold,                    \
        T_PREC const rLowerValue,                   \
        T_PREC const rUpperValue                    \
    );
    INSTANTIATE_cudaKernelCutOff( float )
    INSTANTIATE_cudaKernelCutOff( double )


} // namespace cuda
} // namespace algorithms
} // namespace imresh
