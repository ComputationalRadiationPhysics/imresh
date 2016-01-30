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

#include <algorithm>  // max
#include <cmath>
#ifdef USE_FFTW
#   include <fftw3.h>
#endif


namespace imresh
{
namespace algorithms
{


    template< class T_PREC, class T_COMPLEX >
    void complexNormElementwise
    (
        T_PREC          * const __restrict__ rDataTarget,
        T_COMPLEX const * const __restrict__ rDataSource,
        unsigned int const rnData
    )
    {
        #pragma omp parallel for
        for ( auto i = 0u; i < rnData; ++i )
        {
            auto const re = rDataSource[i][0];
            auto const im = rDataSource[i][1];
            rDataTarget[i] = std::sqrt( re*re + im*im );
        }
    }


    template< class T_COMPLEX, class T_PREC >
    void applyComplexModulus
    (
        T_COMPLEX       * const __restrict__ rData,
        T_PREC    const * const __restrict__ rComplexModulus,
        unsigned int const rnData
    )
    {
        #pragma omp parallel for
        for ( auto i = 0u; i < rnData; ++i )
        {
            auto const re = rData[i][0];
            auto const im = rData[i][1];

            auto norm = std::sqrt( re*re + im*im );
            if ( norm == 0 )
                norm = 1;

            auto const factor = rComplexModulus[i] / norm;
            rData[i][0] = re * factor;
            rData[i][1] = im * factor;
        }
    }

    /* explicitely instantiate needed data types */
    #ifdef USE_FFTW
        template void complexNormElementwise<float,fftwf_complex>
        (
            float               * const __restrict__ rDataTarget,
            fftwf_complex const * const __restrict__ rDataSource,
            unsigned int const rnData
        );
        template void complexNormElementwise<double,fftw_complex>
        (
            double             * const __restrict__ rDataTarget,
            fftw_complex const * const __restrict__ rDataSource,
            unsigned int const rnData
        );

        template void applyComplexModulus<fftwf_complex,float>
        (
            fftwf_complex      * const __restrict__ rData,
            float        const * const __restrict__ rComplexModulus,
            unsigned int const rnData
        );
        template void applyComplexModulus<fftw_complex,double>
        (
            fftw_complex       * const __restrict__ rData,
            double       const * const __restrict__ rComplexModulus,
            unsigned int const rnData
        );
    #endif


} // namespace algorithms
} // namespace imresh
