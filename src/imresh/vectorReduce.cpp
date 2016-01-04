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


#include "vectorReduce.h"


namespace imresh
{


    template<class T>
    T vectorMaxAbsDiff
    (
        T * const & rData1,
        T * const & rData2,
        const unsigned & rnData
    )
    {
        T maxAbsDiff = T(0);
        #pragma omp parallel for reduction( max : maxAbsDiff )
        for ( unsigned i = 0; i < rnData; ++i )
            maxAbsDiff = std::max( maxAbsDiff, std::abs( rData1[i]-rData2[i] ) );
        return maxAbsDiff;
    }

    template<class T>
    T vectorMaxAbs
    (
        T * const & rData,
        const unsigned & rnData
    )
    {
        T maximum = T(0);
        #pragma omp parallel for reduction( max : maximum )
        for ( unsigned i = 0; i < rnData; ++i )
            maximum = std::max( maximum, std::abs( rData[i] ) );
        return maximum;
    }

    template<class T>
    T vectorMax
    (
        T * const & rData,
        const unsigned & rnData
    )
    {
        T maximum = T(0);
        #pragma omp parallel for reduction( max : maximum )
        for ( unsigned i = 0; i < rnData; ++i )
            maximum = std::max( maximum, rData[i] );
        return maximum;
    }

    template<class T>
    T vectorMin
    (
        T * const & rData,
        const unsigned & rnData
    )
    {
        T minimum = T(0);
        #pragma omp parallel for reduction( min : minimum )
        for ( unsigned i = 0; i < rnData; ++i )
            minimum = std::min( minimum, rData[i] );
        return minimum;
    }


    /* explicitely instantiate needed data types */
    template<class T> void dummyFunction( void )
    {
        vectorMaxAbsDiff<T>( NULL, NULL, 0 );
        vectorMaxAbs<T>( NULL, 0 );
        vectorMax<T>( NULL, 0 );
        vectorMin<T>( NULL, 0 );
    }
    template void dummyFunction<float>();
    template void dummyFunction<double>();


} // namespace imresh
