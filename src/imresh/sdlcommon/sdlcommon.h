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
#include <cassert>
#include <cmath>    // sqrt
#include <cstddef>  // NULL
#include <stack>
#include <ctime>    // time, strftime
#include <cstdint>  // uint32_t
#include <SDL.h>
#include <SDL_ttf.h>

#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif


namespace imresh
{
namespace sdlcommon
{


    void SDL_check
    (
        const void * const & rpPointerToCheck,
        const char * const & rFilename,
        const unsigned & rLineNumber
    );

    void SDL_check
    (
        const int & rErrorCode,
        const char * const & rFilename,
        const unsigned & rLineNumber
    );

    #define SDL_CHECK(X) SDL_check(X,__FILE__,__LINE__);

    void checkTTFPtr
    (
        const void * const & rPointerToCheck,
        const char * const & rFileName,
        const unsigned & rLineNumber
    );

    #define CHECK_TTF_PTR(X) checkTTFPtr(X,__FILE__,__LINE__);


    std::ostream & operator<<( std::ostream & rOut, const SDL_Rect & rRect );


    int SDL_SaveRenderedToBmp
    (
        SDL_Renderer * const & rpRenderer,
        SDL_Window * const & rpWindow,
        const char * const & rFilename
    );

    template<class T>
    struct Point2Dx
    {
        T x,y;
        Point2Dx   operator- ( Point2Dx const & b ) const;
        Point2Dx   operator+ ( Point2Dx const & b ) const;
        Point2Dx   operator* ( T        const & b ) const;
        Point2Dx & operator/=( Point2Dx const & b );
        Point2Dx & operator/=( T        const & b );
        T norm(void) const;
    };

    typedef Point2Dx<int  > Point2D;
    typedef Point2Dx<float> Point2Df;

    typedef struct { Point2D  p[2]; } Line2D;
    typedef struct { Point2Df p[2]; } Line2Df;

    template<class T>
    std::ostream & operator<<
    (
        std::ostream & rOut,
        const Point2Dx<T> & rPointToPrint
    );

    std::ostream & operator<<
    (
        std::ostream & out,
        const Line2D & rLineToPrint
    );

    /**
     * calculates the intersection point of two lines
     *
     * @param[out] rIntersectionPoint will hold the intersection point of
     *             rLine0 and rLine1
     **/
    void calcIntersectPoint2Lines
    (
        Line2Df const & rLine0,
        Line2Df const & rLine1,
        Point2Df * const & rIntersectionPoint
    );

    /**
     * @see calcIntersectPoint2Lines
     **/
    void calcIntersectPoint2Lines
    (
        Line2D const & rLine0,
        Line2D const & rLine1,
        Point2Df * const & rIntersectionPoint
    );

    /* draws filled circle. cx and cy are in the center of the specified pixel,
     * meaning r<=0.5 will result in only 1px set */
    void SDL_RenderDrawCircle
    (
        SDL_Renderer * const & rpRenderer,
        const float & rCenterX,
        const float & rCenterY,
        const float & rRadius,
        const bool & rFill = true
    );

    /**
     * Draws a rectangle with thick borders. The borders will always be thickened
     * to the inside. Meaning if rRect specifies 10,10,5,5 and a thickness of 2
     * then the resulting rectangle will consist of 4*4+4*3 pixels belonging
     * to the border and 1 pixel which is empty
     **/
    void SDL_RenderDrawThickRect
    (
        SDL_Renderer * const & rpRenderer,
        const SDL_Rect & rRect,
        const unsigned & rWidth = 1
    );

    void SDL_RenderDrawThickRect
    (
        SDL_Renderer * const & rpRenderer,
        const SDL_Rect * const & rRect,
        const unsigned & rWidth = 1
    );

    /**
     * Draw a line from x1,y1 to x2,y2 with an arrow head at x2,y2
     *
     * @verbatim
     *                                          .
     *                     O                     '.
     *                 --+++                 rAngle'.
     *                / ++ |__     ------------------>   ^
     *                 ++                          .'    | ds
     *                 | \                       .'      |
     *               --+ |                      '        v
     *                                          <---->
     *                                         rHeadSize
     * @endverbatim
     *
     * @param[in] rHeadSize projected length on drawn line to be used for the
     *            arrow head. if rHeadSize == sqrt( (x2-y1)**2 + (y2-y1)**2 )
     *            then the arrow will look like a triangle with its altitue drawn
     * @param[in] rAngle arrow head angle in degrees
     **/
    int SDL_RenderDrawArrow
    (
        SDL_Renderer * const & rpRenderer,
        const int & x1,
        const int & y1,
        const int & x2,
        const int & y2,
        const float & rHeadSize = 5,
        const float & rAngle = 30
    );

    /**
     * parses basic controls like window quit event
     **/
    int SDL_basicControl
    (
        SDL_Event const & event,
        SDL_Window * const & rpWindow,
        SDL_Renderer * const & rpRenderer
    );

    /**
     * Provides control for SDL, like pause/stop, make screenshot, next Frame, ...
     *
     * @param[in]  event SDL event which we shall parse for control keystrokes
     * @param[out] anim a class which provides methods like togglePause,
     *             getCurrentFrameNumber
     * @return 1 if render touched and needs to be redrawn, else 0
     **/
    int SDL_animControl( SDL_Event const & event );

    /* @todo: make these two work if different renderers are given! */
    /* saves the current rendering color of rpRenderer */
    int SDL_RenderPushColor( SDL_Renderer * const & rpRenderer );
    /* restore the last saved rendering color of rpRenderer */
    int SDL_RenderPopColor ( SDL_Renderer * const & rpRenderer );


} // namespace sdlcommon
} // namespace imresh
