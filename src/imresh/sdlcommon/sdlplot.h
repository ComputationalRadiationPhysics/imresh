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
#include <SDL.h>
#include "sdlcommon.h"
#include <cmath>    // log10
#include <cstddef>  // NULL
#include <cfloat>   // FLT_MAX, FLT_MIN


namespace imresh
{
namespace sdlcommon
{


    /**
     * Renders a text string to the given renderer
     *
     * @param[in]  rpRenderer renderer to use
     * @param[in]  rFont font to use
     * @param[in]  rTarget location where to print. If w or h are 0 they are
     *             filled in to match the width and height of the rendered
     *             string, else the texture will be rendered scaled
     * @param[in]  rXAlign 0 if normal left alignment, 1 if central alignment,
     *             2 if right alignment, meaning right side of string to render
     *             touches rTarget.x,y
     * @param[in]  rYAlign vertical alignment, @see rXAlign
     * @param[out] rTarget will be changed if necessary for align or if h or w
     *             were set to 0.
     **/
    int SDL_drawString
    (
        SDL_Renderer * const & rpRenderer,
        TTF_Font * const & rFont,
        const char * const & rStringToDraw,
        SDL_Rect * const & rTarget,
        const int & rXAlign,
        const int & rYAlign
    );

    /**
     * Choose ticks in such a way, that at least 10 ticks are shown
     **/
    float chooseTickSpacing
    (
        const float & a,
        const float & b
    );

    int printPrettyFloat
    (
        char * const & rString,
        const float & rValue
    );

    /**
     * Draw a vertical axis for a plot labeled with values
     *
     * If automatically chooses ticks in such a way, that at least 10 ticks are
     * shown.
     *
     * @param[in] rpRenderer renderer to draw on
     * @param[in] x position of origo, meaning the lower bound of the axis
     * @param[in] y position of origo, note that in the SDL coordinate system
     *            used here larger y values mean more to the bottom in the
     *            window
     * @param[in] h height of the axis in pixels
     * @param[in] y0 lower value bound of the axis, used for drawing labels
     * @param[in] y1 upper value bound of the axis
     **/
    int SDL_RenderDrawAxes
    (
        SDL_Renderer * const & rpRenderer,
        const SDL_Rect & rAxes,
        const float & x0,
        const float & x1,
        const float & y0,
        const float & y1
    );

    /**
     * Wrapper for SDL_RenderDrawAxes using SDL_Rect instead of SDL_Rect*
     **/
    int SDL_RenderDrawAxes
    (
        SDL_Renderer * const & rpRenderer,
        const SDL_Rect * const & rAxes,
        const float & x0,
        const float & x1,
        const float & y0,
        const float & y1
    );

    /**
     * Draws a histogram using values for the bin heights
     *
     * @param[in] rAxes window in which we may draw the plot
     * @param[in] x0,x1 x-axis limits. If x0=x1, then the x-range will be chose
     *            to be [0,nValues-1].
     * @param[in] y0,y1 y-axis limits. If y0==y1, then the y-range will be chosen
     *            automatically to fit all the values.
     * @param[in] values values of the bins in internal coordinates (not in pixel)
     * @param[in] nValues length of values array
     * @param[in] binWidth in pixels
     * @param[in] fill fills bin rectangles or not
     **/
    int SDL_RenderDrawHistogram
    (
        SDL_Renderer * const rpRenderer,
        const SDL_Rect & rAxes,
        float x0,
        float x1,
        float y0,
        float y1,
        const float * const values,
        const unsigned & nValues,
        unsigned binWidth = 0,
        const bool & fill = false,
        const bool & drawAxis = false,
        const char * const & title = ""
    );

    /**
     * Plots a 2D matrix as a pixelated "image"
     *
     * @param[in] rAxes window in which we may draw the plot
     * @param[in] x0,x1 x-axis limits. If x0=x1, then the x-range will be chose
     *            to be [0,nValuesX-1]
     * @param[in] y0,y1 y-axis limits
     * @param[in] values values of the pixels to draw. 2D array in the form
     *            values[iy*nx+ix], meaning the first nx values correspond to
     *            the first line of pixels at y=y0, meaning at the bottom of
     *            the axis/plot. values must be in [0,1], where 0 are drawn black
     *            and 1 are dran white. Higher and smaller values will be drawn red
     * @param[in] nValuesX length of first and subsequent horizontal lines of values
     * @param[in] nValuesY number of lines in values. values must be at least
     *            nValuesX*nValuesY*sizeof(T_PREC) large
     **/
    template<class T_PREC>
    int SDL_RenderDrawMatrix
    (
        SDL_Renderer * const & rpRenderer,
        const SDL_Rect & rAxes,
        float x0,
        float x1,
        float y0,
        float y1,
        T_PREC * const & values,
        const unsigned & nValuesX,
        const unsigned & nValuesY,
        const bool & drawAxis = false,
        const char * const & title = "",
        const bool & useColors = false
    );



    /******************* definitions for sdlplot.tpp ********************/



    /**
     * Plot the function f over some interval [x0,x1] in the plot range [y0,y1]
     *
     * @param[in] rAxes rectangle/window in which the plot is drawn
     **/
    template<class T_FUNC>
    int SDL_RenderDrawFunction
    (
        SDL_Renderer * const & rpRenderer,
        const SDL_Rect & rAxes,
        const float & x0,
        const float & x1,
        const float & y0,
        const float & y1,
        T_FUNC f, const bool & drawAxis = false
    );


} // namespace sdlcommon
} // namespace imresh


#include "sdlplot.tpp"