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

#include <iostream>
#include <vector>
#include <cmath>     // sqrtf
#include <cstring>   // memset
#include <cfloat>    // FLT_EPSILON
#include <fftw3.h>
#include "vectorIndex.h"


namespace imresh
{
namespace algorithms
{

    /**
     * shifts an n-dimensional index by half the size
     *
     * The use case is the FFT where the 0 frequency is at the 0th array entry
     * but we want it in the center, in all dimensions.
     *
     * @param[in] rLinearIndex simple linear index which should be in
     *            [0,product(rDim))
     * @param[in] rDim the size of each dimension
     **/
    unsigned fftShiftIndex
    (
        const unsigned & rLinearIndex,
        const std::vector<unsigned> & rDim
    );

    /**
     * Calculates the diffraction intensity of an object function
     *
     * @see https://en.wikipedia.org/wiki/Diffraction#General_aperture
     * Because there are different similar applications no constant physical
     * factors will be applied here, instead a simple fourier transform
     * followed by a squared norm will be used.
     *
     * This function also shifts the frequency, so that frequency 0 is in the
     * middle like it would be for a normal measurement.
     *
     * E.g. a rectangular box becomes a kind of checkerboard pattern with
     * decreasing maxima intensity:
     *
     * @verbatim
     *        +------------+      +------------+          +------------+
     *        |            |      |78 ++  ++ 56|          |     --     |
     *        |            |      |o> ''  '' <o|          | .. <oo> .. |
     *        |     #      |  FT  |-          -| fftshift | ++ 1234 ++ |
     *        |     #      |  ->  |-          -|  ----->  | ++ 5678 ++ |
     *        |            |      |o> ..  .. <o|          | '' <oo> '' |
     *        |            |      |34 ++  ++ 12|          |     --     |
     *        +------------+      +------------+          +------------+
     *                           k=0         k=N-1              k=0
     * @endverbatim
     *
     * @param[in]  rIoData real valued object which is to be transformed
     * @param[out] rIoData real valued diffraction intensity
     * @param[in]  rDimensions vector containing the dimensions of rIoData.
     *             the number of dimensions i.e. the size of the vector can be
     *             anything > 0
     **/
    void diffractionIntensity
    (
        float * const & rIoData,
        const std::vector<unsigned> & rDim
    );


} // namespace algorithms
} // namespace imresh
