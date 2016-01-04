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
#include <fftw3.h>


namespace imresh
{


    /**
     * Calculate the maximum absolute difference between to arrays
     *
     * Useful for comparing two vectors of floating point numbers
     **/
    template<class T>
    T vectorMaxAbsDiff
    (
        T * const & rData1,
        T * const & rData2,
        const unsigned & rnData
    );

    template<class T>
    T vectorMaxAbs
    (
        T * const & rData,
        const unsigned & rnData
    );

    template<class T>
    T vectorMax
    (
        T * const & rData,
        const unsigned & rnData
    );

    template<class T>
    T vectorMin
    (
        T * const & rData,
        const unsigned & rnData
    );


} // namespace imresh
