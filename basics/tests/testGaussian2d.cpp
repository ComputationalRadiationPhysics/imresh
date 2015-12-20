
#include <SDL.h>
#include <cstdlib> // srand, rand, RAND_MAX
#include <cassert>
#include <cstdio>  // sprintf
#include <cmath>
#include "sdl/sdlplot.h"
#include "math/image/gaussian.h"

#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif


using namespace sdlcommon;
using namespace imresh::math::image;


namespace imresh {
namespace test {


void testGaussianBlur2d
( SDL_Renderer * rpRenderer, SDL_Rect rect, float * data,
  int nDataX, int nDataY, const float sigma, const char * title )
{
    using namespace imresh::math::image;

    char title2[128];
    SDL_RenderDrawMatrix( rpRenderer, rect, 0,0,0,0, data,nDataX,nDataY,
                          true/*drawAxis*/, title );

    SDL_RenderDrawArrow( rpRenderer, rect.x+1.1*rect.w,rect.y+rect.h/2,
                                     rect.x+1.3*rect.w,rect.y+rect.h/2 );
    rect.x += 1.5*rect.w;
    gaussianBlurHorizontal( data, nDataX, nDataY, sigma );
    sprintf( title2,"G_h(s=%0.1f)*%s",sigma,title );
    SDL_RenderDrawMatrix( rpRenderer, rect, 0,0,0,0, data,nDataX,nDataY,
                          true/*drawAxis*/, title2 );

    SDL_RenderDrawArrow( rpRenderer, rect.x+1.1*rect.w,rect.y+rect.h/2,
                                     rect.x+1.3*rect.w,rect.y+rect.h/2 );
    rect.x += 1.5*rect.w;
    gaussianBlurVertical( data, nDataX, nDataY, sigma );
    sprintf( title2,"G_v o G_h(s=%0.1f)*%s",sigma,title );
    SDL_RenderDrawMatrix( rpRenderer, rect, 0,0,0,0, data,nDataX,nDataY,
                          true/*drawAxis*/, title2 );
}

void testGaussian2d( SDL_Renderer * rpRenderer )
{
    /* ideal window size for this test is 1024x640 px */
{
    srand(165158631);
    const int nDataX = 20;
    const int nDataY = 20;
    SDL_Rect rect = { 40,40,5*nDataX,5*nDataY };
    float data[nDataX*nDataY];

    /* Try different data sets */
    /**
     * +--------+        +--------+   # - black
     * |        |        |     .  |   x - gray
     * |     #  |        |p   .i. |   p - lighter gray
     * |#       |   ->   |xo   .  |   o - very light gray
     * |        |        |p   o   |   i - also light gray
     * |    #   |        |   pxp  |   . - gray/white barely visible
     * +--------+        +--------+     - white
     * Note that the two dots at the borders must result in the exact same
     * blurred value (only rotated by 90°). This is not obvious, because
     * we apply gausian first horizontal, then vertical, but it works and
     * makes the gaussian-N-dimensional algorithm much faster!
     **/
    for ( int i = 0; i < nDataX*nDataY; ++i )
        data[i] = 1.0;
    data[10]           = 0;
    data[10*nDataX]    = 0;
    data[12*nDataX+12] = 0;
    testGaussianBlur2d( rpRenderer,rect, data,nDataX,nDataY, 1.0, "3-Points" );
    rect.y += 140;
    assert( data[9] != 1.0 );
    assert( data[9] == data[11] );
    assert( data[9*nDataX] == data[11*nDataX] );
    assert( data[9] == data[11*nDataX] );
    assert( data[nDataX+10] == data[10*nDataX+1] );

    /* same as above in white on black */
    for ( int i = 0; i < nDataX*nDataY; ++i )
        data[i] = 0;
    data[10]           = 1;
    data[10*nDataX]    = 1;
    data[12*nDataX+12] = 1;
    testGaussianBlur2d( rpRenderer,rect, data,nDataX,nDataY, 1.0, "3-Points" );
    rect.y += 140;
    assert( data[9] != 0 );
    assert( data[9] == data[11] );
    assert( data[9*nDataX] == data[11*nDataX] );
    assert( data[9] == data[11*nDataX] );
    assert( data[nDataX+10] == data[10*nDataX+1] );

    /* blur a random image (draw result to the right of above images) */
    rect.x += (3*1.5+1)*(5*nDataX);
    rect.y  = 20;
    for ( int i = 0; i < nDataX*nDataY; ++i )
        data[i] = rand()/(double)RAND_MAX;
    testGaussianBlur2d( rpRenderer,rect, data,nDataX,nDataY, 1.0, "Random" );
    rect.y += 140;
    for ( int i = 0; i < nDataX*nDataY; ++i )
        data[i] = rand()/(double)RAND_MAX;
    testGaussianBlur2d( rpRenderer,rect, data,nDataX,nDataY, 2.0, "Random" );
    rect.y += 140;
}
{
    /* try with quite a large image! */
    srand(165158631);
    const int nDataX = 240;
    const int nDataY = 240;
    SDL_Rect rect = { 30,320,nDataX,nDataY };
    float data[nDataX*nDataY];

    /* fill with random data */
    for ( int i = 0; i < nDataX*nDataY; ++i )
        data[i] = rand()/(double)RAND_MAX;
    testGaussianBlur2d( rpRenderer,rect, data,nDataX,nDataY, 3.0, "Random" );
}
}


} // namespace imresh
} // namespace test
