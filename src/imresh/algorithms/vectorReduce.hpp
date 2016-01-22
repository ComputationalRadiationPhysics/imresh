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

#include <algorithm>  // max
#include <cmath>
#include <limits>     // lowest, max
#include <cassert>


namespace imresh
{
namespace algorithms
{


    /**
     * Calculate the maximum absolute difference between to arrays
     *
     * Useful for comparing two vectors of floating point numbers
     **/
    template<class T>
    T vectorMaxAbsDiff
    (
        const T * const & rData1,
        const T * const & rData2,
        const unsigned & rnData,
        const unsigned & rnStride = 1
    );

    template<class T>
    T vectorMaxAbs
    (
        const T * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride = 1
    );

    template<class T>
    T vectorMax
    (
        const T * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride = 1
    );

    template<class T>
    T vectorMin
    (
        const T * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride = 1
    );

    template<class T>
    T vectorSum
    (
        const T * const & rData,
        const unsigned & rnData,
        const unsigned & rnStride = 1
    );

} // namespace algorithms
} // namespace imresh
