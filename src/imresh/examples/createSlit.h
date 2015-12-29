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

#include <cstring>  // memset
#include <cmath>    // ceilf


namespace imresh
{
namespace examples
{


    /**
     * Create a sample data of a rectangular object valued 1.0
     *
     * @verbatim
     *    +------+
     *    |      |
     *    |  ##  |
     *    |      |
     *    +------+
     * @endverbatim
     *
     * @param[in] Nx width of the test image to produce
     * @param[in] Ny height of the test image to produce
     * @return pointer to allocated data. Must be deallocated with delete[]
     **/
    float * createVerticalSingleSlit
    (
        const unsigned & Nx,
        const unsigned & Ny
    );


} // namespace examples
} // namespace imresh
