
#include <SDL.h>
#include <cstdlib> // srand, rand, RAND_MAX
#include <cassert>
#include <cstdio>  // sprintf
#include <cmath>
#include "sdlplot.h"
#include "gaussian.h"

#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif

using namespace sdlcommon;


namespace imresh {
namespace test {


void testGaussianBlurVector
( SDL_Renderer * rpRenderer, SDL_Rect rect, float * data, int nData,
  const float sigma, const char * title )
{
    using ::imresh::gaussianblur::gaussianBlur;

    SDL_RenderDrawHistogram(rpRenderer, rect, 0,0,0,0,
        data,nData, 0/*binWidth*/,false/*fill*/, true/*drawAxis*/, title );
    SDL_RenderDrawArrow( rpRenderer, rect.x+1.1*rect.w,rect.y+rect.h/2,
                                     rect.x+1.3*rect.w,rect.y+rect.h/2 );
    rect.x += 1.5*rect.w;

    gaussianBlur( data, nData, sigma );

    char title2[128];
    sprintf( title2,"G(s=%0.1f)*%s",sigma,title );
    SDL_RenderDrawHistogram(rpRenderer, rect, 0,0,0,0,
        data,nData, 0/*binWidth*/,false/*fill*/, true/*drawAxis*/, title2 );
}

void testGaussian( SDL_Renderer * rpRenderer )
{
    srand(165158631);
    SDL_Rect rect = { 40,40,200,80 };

    const int nData = 50;
    float data[nData];

    /* Try different data sets */
    for ( int i = 0; i < nData; ++i )
        data[i] = 255*rand()/(double)RAND_MAX;
    testGaussianBlurVector( rpRenderer,rect,data,nData, 1.0, "Random" );
    rect.y += 100;
    for ( int i = 0; i < nData; ++i )
        data[i] = 255*rand()/(double)RAND_MAX;
    testGaussianBlurVector( rpRenderer,rect,data,nData, 2.0, "Random" );
    rect.y += 100;
    for ( int i = 0; i < nData; ++i )
        data[i] = 255*rand()/(double)RAND_MAX;
    testGaussianBlurVector( rpRenderer,rect,data,nData, 4.0, "Random" );
    rect.y += 100;

    for ( int i = 0; i < nData; ++i )
        data[i] = i > nData/2 ? 1 : 0;
    testGaussianBlurVector( rpRenderer,rect,data,nData, 1.0, "Step" );
    rect.y += 100;
    for ( int i = 0; i < nData; ++i )
        data[i] = i > nData/2 ? 1 : 0;
    testGaussianBlurVector( rpRenderer,rect,data,nData, 4.0, "Step" );
    rect.y += 100;

    {
    const int nData2 = 100;
    float data2[nData2];
    float sigma = 8.0;
    float a =  1.0/( sqrt(2.0*M_PI)*sigma );
    float b = -1.0/( 2.0*sigma*sigma );
    for ( int i = 0; i < nData2; ++i )
        data2[i] = a*exp( (i-nData2/2)*(i-nData2/2)*b );
    char title[64];
    sprintf(title,"G(s=%.2f)",sigma);

    /* a guassian with @f[ \mu_1, \sigma_1 @f] convoluted with a gaussian
     * with @f[ \mu_1, \sigma_1 @f] results also in a gaussian with
     * @f[ \mu = \mu_1+\mu_2, \sigma = \sqrt{ \sigma_1^2+\sigma_2^2 } @f] */
    testGaussianBlurVector( rpRenderer,rect,data2,nData2, sigma, title );
    rect.y += 100;
    }
}

} // namespace imresh
} // namespace test
