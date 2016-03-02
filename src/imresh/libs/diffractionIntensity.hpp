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

#include <utility>
#ifndef USE_FFTW
#   include <cuda_to_cupla.hpp>
#endif


namespace imresh
{
namespace libs
{


#if USE_FFTW

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
        float * rIoData,
        unsigned int rImageWidth,
        unsigned int rImageHeight
    );

#else

    /**
     * @see diffractionIntensity
     *
     * If USE_FFTW is not specified, then this version is used, which uses
     * cufft instead
     **/
    void diffractionIntensity
    (
        float * rIoData,
        unsigned int rImageWidth,
        unsigned int rImageHeight,
        cudaStream_t rStream = 0,
        bool rAsync = false
    );

#endif


} // namespace libs
} // namespace imresh
