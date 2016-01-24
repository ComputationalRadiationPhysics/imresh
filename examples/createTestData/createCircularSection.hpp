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

#include <cmath>    // atan2
#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif


namespace examples
{
namespace createTestData
{


    /**
     * Create a sample data of a circular section with values 1.0 else 0.0
     *
     * By default, if only 3 parameters given a full circle will be drawn
     *
     * @param[in] Nx width  of the test image to produce
     * @param[in] Ny height of the test image to produce
     * @param[in] r width  of the slit (percentage of Nx) 0 <= Dx <= 1
     * @param[in] x0 position of circle in relativ coordinates
     * @param[in] y0 position of circle in relativ coordinates
     * @param[in] Dy height of the slit (percentage of Ny) 0 <= Dy <= 1
     * @param[in] phi can be used to rotate the whole rectangle
     * @return pointer to allocated data. Must be deallocated with delete[]
     **/
    float * createCircularSection
    (
        unsigned const & Nx,
        unsigned const & Ny,
        float    const & r,
        float    const & x0 = 0.5,
        float    const & y0 = 0.5,
        float    const & phi0 = 0,
        float    const & phi1 = 2.0*M_PI
    );


} // namespace createTestData
} // namespace examples
