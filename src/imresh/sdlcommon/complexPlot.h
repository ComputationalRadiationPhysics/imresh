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

#include <cmath>    // log10
#include <complex>
#include <SDL.h>
#include <fftw3.h>
#include "sdlplot.h"
#include "colors/conversions.h"


namespace imresh
{
namespace sdlcommon
{

    /**
     * Uses domain coloring to plot a complex valued matrix
     *
     * The matrix could e.g. contain evaluations of a complex function.
     *
     * @param[in] logPlot if true, then instead of the magnitude of the complex
     *            number log(|z|) will be plotted.
     * @param[in] swapQuadrants true: rows and columns will be shifted by half
     *            width thereby centering the shortest wavelengths instead of
     *            those being at the corners
     * @param[in] colorFunction 1:HSL (H=arg(z), S=1, L=|z|)
     *                          2:HSV
     *                          3:
     * @see SDL_RenderDrawMatrix for other parameters
     **/
    void SDL_RenderDrawComplexMatrix
    (
        SDL_Renderer * const & rpRenderer,
        const SDL_Rect & rAxes,
        const float & x0,
        const float & x1,
        const float & y0,
        const float & y1,
        fftw_complex * const & values,
        const unsigned & nValuesX,
        const unsigned & nValuesY,
        const bool & drawAxis,
        const char * const & title,
        const bool & logPlot = true,
        const bool & swapQuadrants = false,
        const int & colorFunction = 2
    );


} // namespace sdlcommon
} // namespace imresh
