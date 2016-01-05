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


#include "vectorReduce.hpp"


namespace imresh
{
namespace algorithms
{


    template< class T_PREC, class T_COMPLEX >
    void complexNormElementwise
    (
        T_PREC * const & rDataTarget,
        const T_COMPLEX * const & rDataSource,
        const unsigned & rnData
    )
    {
        #pragma omp parallel for
        for ( unsigned i = 0; i < rnData; ++i )
        {
            const float & re = rDataSource[i][0];
            const float & im = rDataSource[i][1];
            rDataTarget[i] = sqrtf( re*re + im*im );
        }
    }


    template< class T_COMPLEX, class T_PREC >
    void applyComplexModulus
    (
        T_COMPLEX * const & rDataTarget,
        const T_COMPLEX * const & rDataSource,
        const T_PREC * const & rComplexModulus,
        const unsigned & rnData
    )
    {
        #pragma omp parallel for
        for ( unsigned i = 0; i < rnData; ++i )
        {
            const auto & re = rDataSource[i][0];
            const auto & im = rDataSource[i][1];
            auto norm = sqrtf(re*re+im*im);
            if ( norm == 0 )
                norm = 1;
            const float factor = rComplexModulus[i] / norm;
            rDataTarget[i][0] = re * factor;
            rDataTarget[i][1] = im * factor;
        }
    }

    /* explicitely instantiate needed data types */
    template< class T_PREC, class T_COMPLEX > void __df1( void )
    {
        complexNormElementwise<T_PREC,T_COMPLEX>( NULL, NULL, 0 );
        applyComplexModulus<T_COMPLEX,T_PREC>( NULL, NULL, NULL, 0 );
    }
    template<class T_COMPLEX> void __df0( void )
    {
        __df1<float ,T_COMPLEX>();
        __df1<double,T_COMPLEX>();
    }
    template void __df0<fftw_complex >();
    template void __df0<fftwf_complex>();


} // namespace algorithms
} // namespace imresh
