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


#include "vectorElementwise.hpp"
#include "vectorElementwise.tpp"


namespace imresh
{
namespace algorithms
{


    /* explicit instantiations */

    #include "libs/alpaka_T_ACC.hpp"
    #define inline

    #define INSTANTIATE_TMP( T_COMPLEX, T_PREC )            \
    template                                                \
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY void                  \
    cudaKernelApplyHioDomainConstraints<T_COMPLEX, T_PREC>  \
    ::template operator()<T_ACC>                            \
    (                                                       \
        T_ACC const & acc,                                  \
        T_COMPLEX       * const rdpgPrevious,               \
        T_COMPLEX const * const rdpgPrime,                  \
        T_PREC    const * const rdpIsMasked,                \
        unsigned int const rnElements,                      \
        T_PREC const rHioBeta                               \
    ) const;
    INSTANTIATE_TMP( cufftComplex, float )
    #undef INSTANTIATE_TMP

    template
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY
    void
    cudaKernelCopyToRealPart<cufftComplex,float>
    ::template operator()<T_ACC>
    (
        T_ACC const & acc,
        cufftComplex * const rTargetComplexArray,
        float * const rSourceRealArray,
        unsigned int const rnElements
    ) const;

    template
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY void
    cudaKernelCopyFromRealPart<float,cufftComplex>
    ::template operator()<T_ACC>
    (
        T_ACC const & acc,
        float * const rTargetComplexArray,
        cufftComplex * const rSourceRealArray,
        unsigned int const rnElements
    ) const;


    #define INSTANTIATE_TMP( T_PREC, T_COMPLEX )        \
    template                                            \
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY void              \
    cudaKernelComplexNormElementwise<T_PREC,T_COMPLEX>  \
    ::template operator()<T_ACC>                        \
    (                                                   \
        T_ACC const & acc,                              \
        T_PREC * const rdpDataTarget,                   \
        T_COMPLEX const * const rdpDataSource,          \
        unsigned int const rnElements                   \
    ) const;
    INSTANTIATE_TMP( float, cufftComplex )
    INSTANTIATE_TMP( cufftComplex, cufftComplex )
    #undef INSTANTIATE_TMP

    #define INSTANTIATE_TMP( T_PREC, T_COMPLEX )        \
    template                                            \
    void cudaComplexNormElementwise<T_PREC, T_COMPLEX>  \
    (                                                   \
        T_PREC * const rdpDataTarget,                   \
        T_COMPLEX const * const rdpDataSource,          \
        unsigned int const rnElements,                  \
        cudaStream_t const rStream,                     \
        bool const rAsync                               \
    );
    INSTANTIATE_TMP( float, cufftComplex )
    INSTANTIATE_TMP( cufftComplex, cufftComplex )
    #undef INSTANTIATE_TMP

    #define INSTANTIATE_TMP( T_COMPLEX, T_PREC )    \
    template                                        \
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY void          \
    cudaKernelApplyComplexModulus<T_COMPLEX,T_PREC> \
    ::template operator()<T_ACC>                    \
    (                                               \
        T_ACC const & acc,                          \
        T_COMPLEX * const rdpDataTarget,            \
        T_COMPLEX const * const rdpDataSource,      \
        T_PREC const * const rdpComplexModulus,     \
        unsigned int const rnElements               \
    ) const;
    INSTANTIATE_TMP( cufftComplex, float )
    #undef INSTANTIATE_TMP

    #define INSTANTIATE_TMP( T_PREC )               \
    template                                        \
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY               \
    void cudaKernelCutOff<T_PREC>                   \
    ::template operator()<T_ACC>                    \
    (                                               \
        T_ACC const & acc,                          \
        T_PREC * const rData,                       \
        unsigned int const rnElements,              \
        T_PREC const rThreshold,                    \
        T_PREC const rLowerValue,                   \
        T_PREC const rUpperValue                    \
    ) const;
    INSTANTIATE_TMP( float )
    INSTANTIATE_TMP( double )
    #undef INSTANTIATE_TMP

    /* this is necessray after including "libs/alapaka_T_ACC.hpp" or else
     * you will run into many errors when trying to use T_ACC as a simple
     * template parameter after this point */
    #undef T_ACC
    #undef inline


} // namespace algorithms
} // namespace imresh
