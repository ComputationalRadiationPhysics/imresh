#pragma once

#include <iostream>
#include <cassert>
#include <SDL.h>

void SDL_check(const void * errorCode, const char * file, int line );
void SDL_check(int          errorCode, const char * file, int line );
#define SDL_CHECK(X) SDL_check(X,__FILE__,__LINE__);

#include <SDL_ttf.h>
void checkTTFPtr(void const * ptr, const char * file, int line );
#define CHECK_TTF_PTR(X) checkTTFPtr(X,__FILE__,__LINE__);


std::ostream & operator<<( std::ostream & out, SDL_Rect rect )
{
    out << "SDL_Rect = { x:"<<rect.x<<", y:"<<rect.y<<", w:"<<rect.w<<", h:"
        << rect.h<<"}";
    return out;
}


int SDL_SaveRenderedToBmp
( SDL_Renderer * rRenderer, SDL_Window* rWindow, const char * rFilename );

#include <cmath> // sqrt
template<class T> struct Point2Dx
{
    T x,y;
    Point2Dx   operator- ( Point2Dx const & b ) const { return Point2Dx{ x-b.x, y-b.y }; }
    Point2Dx   operator+ ( Point2Dx const & b ) const { return Point2Dx{ x+b.x, y+b.y }; }
    Point2Dx   operator* ( T        const & b ) const { return Point2Dx{ x*b, y*b }; }
    Point2Dx & operator/=( Point2Dx const & b ) { x/=b.x; y/=b.y; return *this; }
    Point2Dx & operator/=( T        const & b ) { x/=b; y/=b; return *this; }
    T norm(void) const { return std::sqrt(x*x+y*y); }
};

typedef Point2Dx<int  > Point2D;
typedef Point2Dx<float> Point2Df;

typedef struct { Point2D  p[2]; } Line2D;
typedef struct { Point2Df p[2]; } Line2Df;

template<class T> std::ostream & operator<<( std::ostream & out, Point2Dx<T> p0 )
{ out << "(" << p0.x << "," << p0.y << ")"; return out; }
std::ostream & operator<<( std::ostream & out, Line2D l0 )
{ out << l0.p[0] << "->" << l0.p[1]; return out; }

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
( SDL_Renderer * rpRenderer, const SDL_Rect & rRect, const int rWidth );

void SDL_RenderDrawThickRect
( SDL_Renderer * rpRenderer, const SDL_Rect * rRect, const int rWidth )
{ return SDL_RenderDrawThickRect(rpRenderer,*rRect,rWidth); }



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
struct classSDL_RenderDrawThickLine
{
    /* This is the height of the line covering the pixel squares */
    float f(float x, float a, float b);
    /* f integrated therefor the area of the pixels covered by the line */
    float F(float x, float a, float b);
    void operator()
    ( SDL_Renderer * rpRenderer, int x0, int y0, int x1, int y1, float width );
} SDL_RenderDrawThickLine;


template <class T> int sgn(const T & x) { return ( T(0)<x) - (x<T(0) ); }

void SDL_RenderDrawThickLine2
( SDL_Renderer * rpRenderer, int x0, int y0, int x1, int y1, float width );


int SDL_basicControl(SDL_Event const & event, SDL_Window * rpWindow, SDL_Renderer * rpRenderer );
/**
 * Provides control for SDL, like pause/stop, make screenshot, next Frame, ...
 *
 * @param[in]  event SDL event which we shall parse for control keystrokes
 * @param[out] anim a class which provides methods like togglePause,
 *             getCurrentFrameNumber
 **/
template<class T_ANIMATION>
void basicAnimControl( SDL_Event const & event, T_ANIMATION & anim );

class SDL_drawLineControl {
private:
    Point2D p0;  // last point clicked at
    Line2D  l0;  // last drawn line
public:
    SDL_drawLineControl(void) : p0( Point2D{-1,-1} ), l0( Line2D{p0,p0} ) {};
    int operator()( SDL_Event const & event, SDL_Renderer * rpRenderer );
};

#include "sdlcommon.cpp"
