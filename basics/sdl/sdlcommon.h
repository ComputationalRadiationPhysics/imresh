#pragma once

#include <iostream>
#include <cassert>
#include <cmath>    // sqrt
#include <stack>
#include <ctime>    // time, strftime
#include <cstdint>  // uint32_t
#include <SDL.h>
#include <SDL_ttf.h>

#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif


namespace sdlcommon {


void SDL_check(const void * errorCode, const char * file, int line );
void SDL_check(int          errorCode, const char * file, int line );
#define SDL_CHECK(X) SDL_check(X,__FILE__,__LINE__);

void checkTTFPtr(void const * ptr, const char * file, int line );
#define CHECK_TTF_PTR(X) checkTTFPtr(X,__FILE__,__LINE__);


std::ostream & operator<<( std::ostream & out, SDL_Rect rect );


int SDL_SaveRenderedToBmp
( SDL_Renderer * rRenderer, SDL_Window* rWindow, const char * rFilename );

template<class T> struct Point2Dx
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

template<class T> std::ostream & operator<<( std::ostream & out, Point2Dx<T> p0 );
std::ostream & operator<<( std::ostream & out, Line2D l0 );

void calcIntersectPoint2Lines
( Line2Df const & l0, Line2Df const & l1, Point2Df * intsct );

void calcIntersectPoint2Lines
( Line2D const & l0, Line2D const & l1, Point2Df * intsct );



/* draws filled circle. cx and cy are in the center of the specified pixel,
 * meaning r<=0.5 will result in only 1px set */
void SDL_RenderDrawCircle
( SDL_Renderer * renderer, const float & cx, const float & cy,
  const float & r, bool fill = true );



/**
 * Draws a rectangle with thick borders. The borders will always be thickened
 * to the inside. Meaning if rRect specifies 10,10,5,5 and a thickness of 2
 * then the resulting rectangle will consist of 4*4+4*3 pixels belonging
 * to the border and 1 pixel which is empty
 **/
void SDL_RenderDrawThickRect
( SDL_Renderer * const rpRenderer, const SDL_Rect & rRect, const int rWidth );

void SDL_RenderDrawThickRect
( SDL_Renderer * const rpRenderer, const SDL_Rect * rRect, const int rWidth );



/**
 * Draw a line with a crude arrow head at x2,y2
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
( SDL_Renderer* rpRenderer, int x1, int y1, int x2, int y2,
  float rHeadSize = 5, float rAngle=20 );

/* TODO!!! */
void SDL_RenderDrawThickLine
( SDL_Renderer * const rpRenderer, int x0, int y0, int x1, int y1, float width );


template <class T> int sgn(const T & x) { return ( T(0)<x) - (x<T(0) ); }

void SDL_RenderDrawThickLine2
( SDL_Renderer * const rpRenderer, int x0, int y0, int x1, int y1, float width );


int SDL_basicControl(SDL_Event const & event, SDL_Window * rpWindow, SDL_Renderer * const rpRenderer );

/**
 * Provides control for SDL, like pause/stop, make screenshot, next Frame, ...
 *
 * @param[in]  event SDL event which we shall parse for control keystrokes
 * @param[out] anim a class which provides methods like togglePause,
 *             getCurrentFrameNumber
 * @return 1 if render touched and needs to be redrawn, else 0
 **/
int SDL_animControl( SDL_Event const & event );


class SDL_drawLineControl {
private:
    Point2D p0;  // last point clicked at
    Line2D  l0;  // last drawn line
public:
    SDL_drawLineControl(void) : p0( Point2D{-1,-1} ), l0( Line2D{p0,p0} ) {};
    int operator()( SDL_Event const & event, SDL_Renderer * const rpRenderer );
};

/* @todo: make these two work if different renderers are given! */
/* saves the current rendering color of rpRenderer */
int SDL_RenderPushColor(SDL_Renderer * const rpRenderer);
/* restore the last saved rendering color of rpRenderer */
int SDL_RenderPopColor(SDL_Renderer * const rpRenderer);


} // namespace sdlcommon

#include "sdlcommon.tpp"
