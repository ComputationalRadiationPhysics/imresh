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


#include "sdlcommon.h"


namespace imresh
{
namespace sdlcommon
{


void SDL_check
(
    const int & rErrorCode,
    const char * const & rFilename,
    const int & rLineNumber )
{
    if ( rErrorCode != 0 )
    {
        std::cout
        << "SDL error in " << rFilename << " line:" << rLineNumber << " : "
        << SDL_GetError() << "\n" << std::flush;
        assert(false);
    }
}

void SDL_check
(
    const void * const & rPointerToCheck,
    const char * const & rFilename,
    const int & rLineNumber
)
{
    if ( rPointerToCheck == NULL )
        SDL_check( 1, rFilename, rLineNumber );
}

void checkTTFPtr
(
    void const * const & rPointerToCheck,
    const char * const & rFilename,
    const int & rLineNumber
)
{
    if ( rPointerToCheck == NULL )
    {
        std::cout
        << "TTF error in " << rFilename << " line:" << rLineNumber << " : "
        << TTF_GetError() << "\n";
        assert(false);
    }
}

std::ostream & operator<<
(
    std::ostream & rOut,
    const SDL_Rect & rRect )
{
    rOut << "SDL_Rect = { "
        << "x:" << rRect.x << ", "
        << "y:" << rRect.y << ", "
        << "w:" << rRect.w << ", "
        << "h:" << rRect.h << "}";
    return rOut;
}

int SDL_SaveRenderedToBmp
(
    SDL_Renderer * const & rpRenderer,
    SDL_Window * const & rpWindow,
    const char * const & rFilename
)
{
    int w,h;
    SDL_GetRendererOutputSize( rpRenderer, &w, &h );

    SDL_Surface *saveSurface = SDL_CreateRGBSurface( 0, w, h, 32,
        /* channel bitmasks */ 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000 );
    SDL_RenderReadPixels( rpRenderer, NULL, SDL_PIXELFORMAT_ARGB8888,
        saveSurface->pixels, saveSurface->pitch );
    int errorCode = SDL_SaveBMP( saveSurface,rFilename );
    SDL_FreeSurface(saveSurface);

    return errorCode;
}

template<class T>
Point2Dx<T> Point2Dx<T>::operator-
( Point2Dx const & b ) const
{
    return Point2Dx{ x-b.x, y-b.y };
}

template<class T>
Point2Dx<T> Point2Dx<T>::operator+
( Point2Dx const & b ) const
{
    return Point2Dx{ x+b.x, y+b.y };
}

template<class T>
Point2Dx<T> Point2Dx<T>::operator*
( T const & b ) const
{
    return Point2Dx{ x*b, y*b };
}

template<class T>
Point2Dx<T> & Point2Dx<T>::operator/=
( Point2Dx const & b )
{
    x/=b.x; y/=b.y; return *this;
}

template<class T>
Point2Dx<T> & Point2Dx<T>::operator/=
( T const & b )
{
    x/=b; y/=b; return *this;
}

template<class T>
T Point2Dx<T>::norm( void ) const
{
    return std::sqrt(x*x+y*y);
}

template<class T>
std::ostream & operator<<
(
    std::ostream & rOut,
    const Point2Dx<T> & rPointToPrint
)
{
    rOut << "(" << rPointToPrint.x << "," << rPointToPrint.y << ")";
    return rOut;
}

std::ostream & operator<<
(
    std::ostream & rOut,
    const Line2D & rLineToPrint
)
{
    rOut << rLineToPrint.p[0] << "->" << rLineToPrint.p[1];
    return rOut;
}


/* explicitely instantiating operator<< for the Point2Dx<T> automatically
 * instantiates Point2Dx and all it's methods for those template parameters,
 * thereby saving quite some text to type */
template std::ostream & operator<< <float>( std::ostream & rOut, const Point2Dx<float> & rPointToPrint );
template std::ostream & operator<< <double>( std::ostream & rOut, const Point2Dx<double> & rPointToPrint );
template std::ostream & operator<< <int>( std::ostream & rOut, const Point2Dx<int> & rPointToPrint );


void calcIntersectPoint2Lines
(
    Line2Df const & rLine0,
    Line2Df const & rLine1,
    Point2Df * const & rIntersectionPoint
)
{
    /* Test if they are even intersecting */
    /*Point2Df v,w,r; float s,u;
    v = l0.p[1] - l0.p[0]; v /= v.norm();
    w = l1.p[1] - l1.p[0]; w /= w.norm();
    r = l1.p[0] - l0.p[0];
    u = v.x*w.y - v.y*w.x; // |v cross w|
    s =  ( w.x*r.y - w.y*r.x )/u; // |w cross r|/u
    //t = -( r.x*v.y - r.y*v.x )/u;
    */
    const Point2Df & p0 = rLine0.p[0];
    const Point2Df & p1 = rLine1.p[0];
    Point2Df v0 = rLine0.p[1] - p0;
    Point2Df v1 = rLine1.p[1] - p1;
    const float s =   ( v1.y*(p1.x-p0.x) + (p0.y-p1.y)*v1.x ) /
                            ( v0.x*v1.y  -  v0.y*v1.x );
    const float t = - ( v0.y*(p1.x-p0.x) + (p0.y-p1.y)*v0.x ) /
                            ( v1.x*v0.y  -  v1.y*v0.x );
    /* note: comparison with NaN always false */
    if ( s >= 0 and s <= 1 and t >= 0 and t <= 1 )
        *rIntersectionPoint = p0 + v0*s;
        //*intsct = p1 + v1*t;
}

void calcIntersectPoint2Lines
(
    Line2D const & rLine0,
    Line2D const & rLine1,
    Point2Df * const & rIntersectionPoint
)
{
    const Line2Df l0f = Line2Df{ (float) rLine0.p[0].x, (float) rLine0.p[0].y,
                                 (float) rLine0.p[1].x, (float) rLine0.p[1].y };
    const Line2Df l1f = Line2Df{ (float) rLine1.p[0].x, (float) rLine1.p[0].y,
                                 (float) rLine1.p[1].x, (float) rLine1.p[1].y };
    calcIntersectPoint2Lines( l0f, l1f, rIntersectionPoint );
}

void SDL_RenderDrawCircle
(
    SDL_Renderer * const & rpRrenderer,
    const float & rCenterX,
    const float & rCenterY,
    const float & rRadius,
    const bool & rFill
)
{
    for ( int h=ceil(rRadius); h > -ceil(rRadius); --h )
    {
        const int s = floor(sqrt(rRadius*rRadius-h*h));
        SDL_RenderDrawLine( rpRrenderer,
                            rCenterX-s, rCenterY+h,
                            rCenterX+s, rCenterY+h );
    }
}

void SDL_RenderDrawThickRect
(
    SDL_Renderer * const & rpRenderer,
    const SDL_Rect & rRect,
    const int & rWidth
)
{
    /* It is allowed, that rRect.w < 2*rWidth, but in that case will result
     * in a seemingly filled rectangle */
    assert( rRect.w >= 0 );
    assert( rRect.h >= 0 );

    /* writable copy of argument and short-hand aliases */
    SDL_Rect rect = rRect;
    int &x = rect.x, &y = rect.y, &w = rect.w, &h = rect.h;

    for ( int i = 0; i < rWidth; ++i )
    {
        SDL_RenderDrawRect ( rpRenderer, &rect );

        /* @see https://bugzilla.libsdl.org/show_bug.cgi?id=3182 */
        SDL_Point points[4];
        points[0].x = x;
        points[0].y = y;
        points[1].x = x+w-1;
        points[1].y = y;
        points[2].x = x;
        points[2].y = y+h-1;
        points[3].x = x+w-1;
        points[3].y = y+h-1;
        SDL_RenderDrawPoints( rpRenderer, points, 4 );

        x += 1;
        y += 1;
        w -= 2;
        h -= 2;
        if ( w <= 0 or h <= 0 )
            break;
    }
}

void SDL_RenderDrawThickRect
(
    SDL_Renderer * const rpRenderer,
    const SDL_Rect * const & rRect,
    const int & rWidth
)
{
    return SDL_RenderDrawThickRect(rpRenderer,*rRect,rWidth);
}

int SDL_RenderDrawArrow
(
    SDL_Renderer * const & rpRenderer,
    const int & x1,
    const int & y1,
    const int & x2,
    const int & y2,
    const float & rHeadSize,
    const float & rAngle
)
{
    SDL_RenderDrawLine( rpRenderer, x1,y1, x2,y2 );
    float ds = tan( rAngle * M_PI / 180.0f ) * rHeadSize;

    #if false
    /* for phi=0, meaning the arrow points to the right : dy=ds, dx=rHeadSize
     * if that is not the case, we need to rotate these two */
    float phi = atan( (float)(y2-y1) / (float)(x2-x1) );
    float dxL = -rHeadSize;
    float dxR = -rHeadSize;
    float dyL = +ds; /* y shift for the upper arrow stroke (left  of arrow) */
    float dyR = -ds; /* y shift for the upper arrow stroke (right of arrow) */

    float dxLT =
    #endif

    /* (x1,y1).(x2,y2) != 0 => y2 = x1/y1*x2 => found orthogonal to our line */
    float x,y;
    x = x2-x1;
    y = y2-y1;
    float dsline = sqrt( x*x + y*y );
    x = x/dsline*rHeadSize;
    y = y/dsline*rHeadSize;

    /* vector orthogonal to drawn line */
    float xp,yp;
    if ( y == 0 )
        xp = 0, yp = x;
    else
        xp = 1, yp = -x/y;
    float dsperp = sqrt( xp*xp + yp*yp );
    xp = xp/dsperp*ds;
    yp = yp/dsperp*ds;

    SDL_RenderDrawLine( rpRenderer, x2,y2, x2-x+xp, y2-y+yp );
    SDL_RenderDrawLine( rpRenderer, x2,y2, x2-x-xp, y2-y-yp );

    return 0;
}


const char * getTimeString( void )
{
    time_t date;
    const int nBufferChars = 128;
    static char buffer[nBufferChars];
    struct tm* tm_info;

    time(&date);
    tm_info = localtime(&date);

    strftime(buffer, nBufferChars, "%Y-%m-%d_%H-%M-%S", tm_info);
    return buffer;
}

int SDL_basicControl
(
    SDL_Event const & event,
    SDL_Window * const & rpWindow,
    SDL_Renderer * const & rpRenderer
)
{
    switch ( event.type )
    {
        case SDL_QUIT:
            return 1;
        case SDL_KEYDOWN:
            switch ( event.key.keysym.sym )
            {
                case SDLK_F5:
                {
                    char filename[256];
                    sprintf( filename, "%s.bmp", getTimeString() );
                    int err = SDL_SaveRenderedToBmp( rpRenderer,rpWindow,filename );
                    if (err == 0)
                        std::cout << "Saved screenshot to " << filename << "\n";
                    else
                        std::cout << "Couldn't save "<<filename<<" does the target folder exist and has the correct permissions?\n";
                    break;
                }
                case SDLK_q:
                case SDLK_ESCAPE:
                    return 1;
                default: break;
            }
            break;
        default: break;
    }
    return 0;
}


int SDL_animControl( SDL_Event const & event )
{
    static bool animPaused = true;
    static uint32_t timeLastFrame = 0;  // milliseconds
    static const int renderFPS = 1;  // frames per second
    static uint32_t renderFrameDelay = 1000 / renderFPS; // milliseconds

    switch ( event.type )
    {
        case SDL_KEYDOWN:
            switch ( event.key.keysym.sym )
            {
                case SDLK_s:
                    return 1;
                    break;
                case SDLK_PLUS:
                    renderFrameDelay /= 2;
                    break;
                case SDLK_MINUS:
                    renderFrameDelay *= 2;
                    if ( renderFrameDelay <= 0 )
                        renderFrameDelay = 1;
                    break;
                case SDLK_SPACE:
                    animPaused = not animPaused;
                    break;
                default: break;
            }
            break;
        default: break;
    }

    /* Render next frame if enough time bygone and not paused */
    if ( not animPaused and
         ( SDL_GetTicks() - timeLastFrame ) > renderFrameDelay )
    {
        std::cout << "animation running\n";
        timeLastFrame = SDL_GetTicks();
        return 1; // render touched
    }

    return 0; // render not touched
}


std::stack<SDL_Color> savedRenderingColors;

int SDL_RenderPushColor( SDL_Renderer * const & rpRenderer )
{
    SDL_Color c;
    const int error = SDL_GetRenderDrawColor( rpRenderer, &c.r, &c.g, &c.b, &c.a );
    savedRenderingColors.push(c);
    return error;
}

int SDL_RenderPopColor( SDL_Renderer * const & rpRenderer )
{
    /* test if stack is empty */
    if ( savedRenderingColors.empty() )
        return 1;
    SDL_Color c = savedRenderingColors.top();
    savedRenderingColors.pop();
    return SDL_SetRenderDrawColor( rpRenderer, c.r, c.g, c.b, c.a );
}


} // namespace sdlcommon
} // namespace imresh
