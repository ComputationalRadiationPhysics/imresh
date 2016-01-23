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

#include <algorithm>  // max
#include <cmath>
#include <limits>     // lowest, max
#include <cassert>


namespace imresh
{
namespace algorithms
{


    template<class T_PREC>
    T_PREC vectorMaxAbsDiff
    (
        const T_PREC * const & rData1,
        const T_PREC * const & rData2,
        const unsigned & rnData,
        const unsigned & rnStride
    )
    {
        assert( rnStride > 0 );
        T_PREC maxAbsDiff = T_PREC(0);
        #pragma omp parallel for reduction( max : maxAbsDiff )
        for ( unsigned i = 0; i < rnData*rnStride; i += rnStride )
            maxAbsDiff = std::max( maxAbsDiff, std::abs( rData1[i]-rData2[i] ) );
        return maxAbsDiff;
    }

    template<class T_PREC>
    T_PREC vectorMaxAbs
    (
        const T_PREC * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride
    )
    {
        assert( rnStride > 0 );
        T_PREC maximum = T_PREC(0);
        #pragma omp parallel for reduction( max : maximum )
        for ( unsigned i = 0; i < rnData*rnStride; i += rnStride )
            maximum = std::max( maximum, std::abs( rData[i] ) );
        return maximum;
    }

    template<class T_PREC>
    T_PREC vectorMax
    (
        const T_PREC * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride
    )
    {
        assert( rnStride > 0 );
        T_PREC maximum = std::numeric_limits<T_PREC>::lowest();
        #pragma omp parallel for reduction( max : maximum )
        for ( unsigned i = 0; i < rnData*rnStride; i += rnStride )
            maximum = std::max( maximum, rData[i] );
        return maximum;
    }

    template<class T_PREC>
    T_PREC vectorMin
    (
        const T_PREC * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride
    )
    {
        assert( rnStride > 0 );
        T_PREC minimum = std::numeric_limits<T_PREC>::max();
        #pragma omp parallel for reduction( min : minimum )
        for ( unsigned i = 0; i < rnData*rnStride; i += rnStride )
            minimum = std::min( minimum, rData[i] );
        return minimum;
    }

    template<class T_PREC>
    T_PREC vectorSum
    (
        const T_PREC * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride
    )
    {
        assert( rnStride > 0 );
        T_PREC sum = T_PREC(0);
        #pragma omp parallel for reduction( + : sum )
        for ( unsigned i = 0; i < rnData*rnStride; i += rnStride )
            sum += rData[i];
        return sum;
    }


    /* explicitly instantiate needed data types */

    template float vectorMaxAbsDiff<float>
    (
        const float * const & rData1,
        const float * const & rData2,
        const unsigned & rnData,
        const unsigned & rnStride
    );
    template double vectorMaxAbsDiff<double>
    (
        const double * const & rData1,
        const double * const & rData2,
        const unsigned & rnData,
        const unsigned & rnStride
    );

    template float vectorMaxAbs<float>
    (
        const float * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride
    );
    template double vectorMaxAbs<double>
    (
        const double * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride
    );

    template float vectorMax<float>
    (
        const float * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride
    );
    template double vectorMax<double>
    (
        const double * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride
    );

    template float vectorMin<float>
    (
        const float * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride
    );
    template double vectorMin<double>
    (
        const double * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride
    );

    template float vectorSum<float>
    (
        const float * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride
    );
    template double vectorSum<double>
    (
        const double * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride
    );


} // namespace algorithms
} // namespace imresh
