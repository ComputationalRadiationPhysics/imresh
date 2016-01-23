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

#include <cassert>
#include "rotateCoordinates.hpp"


namespace examples
{
namespace createTestData
{


    /**
     * Create a sample data of a rectangular object valued 1.0
     *
     * @verbatim
     *    +------+ ^
     *    |      | |
     *    |  ##  | Ny   ->  ## Dy
     *    |      | |        Dx
     *    +------+ v
     *    <--Nx-->
     * @endverbatim
     *
     * E.g. Dx = 0.1, Dy=1.0 creates a vertical slit, but simple rectangles
     * can also bec reated with this function
     *
     * @param[in] Nx width  of the test image to produce
     * @param[in] Ny height of the test image to produce
     * @param[in] Dx width  of the slit (percentage of Nx) 0 <= Dx <= 1
     * @param[in] Dy height of the slit (percentage of Ny) 0 <= Dy <= 1
     * @param[in] x0 center of rectangle in relative coordinates
     * @param[in] y0 center of rectangle in relative coordinates
     * @param[in] phi can be used to rotate the whole rectangle
     * @return pointer to allocated data. Must be deallocated with delete[]
     **/
    float * createRectangle
    (
        unsigned const & Nx,
        unsigned const & Ny,
        float    const & Dx = 0.25,
        float    const & Dy = 0.25,
        float    const & x0 = 0.5,
        float    const & y0 = 0.5,
        float    const & phi = 0
    );


} // namespace createTestData
} // namespace examples
