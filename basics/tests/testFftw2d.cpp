
#include <fftw3.h>
#include <cstdio>  // sprintf
#include <cmath>
#include <complex>
#include "sdl/sdlplot.h"

#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif


using namespace sdlcommon;


namespace imresh {
namespace test {


template<class F>
void testFftw2dAndPrint
( SDL_Renderer * rpRenderer, SDL_Rect rAxes, F f, const char* rFuncTitle )
{
    const int nCoefficients = 31;

    fftw_complex *xOriginal, *xTransformed;
    fftw_plan planForward, planInverse;

    xOriginal    = fftw_alloc_complex(nCoefficients);
    xTransformed = fftw_alloc_complex(nCoefficients);
    planForward  = fftw_plan_dft_1d( nCoefficients, xOriginal, xTransformed,
                                     FFTW_FORWARD, FFTW_ESTIMATE);
    planInverse  = fftw_plan_dft_1d( nCoefficients, xTransformed, xOriginal,
                                     FFTW_BACKWARD, FFTW_ESTIMATE);

    float * xReal = new float[nCoefficients];
    float * xImag = new float[nCoefficients];


    /******************* plot original function *******************/
    for ( int i=0; i < nCoefficients; ++i )
    {
        xOriginal[i][0] = f( 2.*M_PI*i/nCoefficients - M_PI );
        xOriginal[i][1] = 0;
    }
    for ( int i=0; i < nCoefficients; ++i )
        xReal[i] = xOriginal[i][0];
    SDL_RenderDrawHistogram( rpRenderer, rAxes, -M_PI,M_PI ,0,0, xReal,
                             nCoefficients, 0,false/*fill*/, true/*drawAxis*/,
                             rFuncTitle );
    SDL_RenderDrawArrow( rpRenderer, rAxes.x+rAxes.w*1.2, rAxes.y+rAxes.h/2,
                                     rAxes.x+rAxes.w*1.4, rAxes.y+rAxes.h/2 );

    /******************* plot coefficients *******************/
    fftw_execute(planForward);
    for ( int i=0; i < nCoefficients; ++i )
    {
        xReal[i] = xTransformed[i][0];
        xImag[i] = xTransformed[i][1];
    }

    char title[128];
    sprintf(title,"Re FT[%s]",rFuncTitle);
    rAxes.x += 2.0*rAxes.w;
    SDL_RenderDrawHistogram(rpRenderer, rAxes, 0,0, 0,0,
        xReal,nCoefficients, 0,false/*fill*/, true/*drawAxis*/, title );

    sprintf(title,"Im FT[%s]",rFuncTitle);
    rAxes.x += 1.5*rAxes.w;
    SDL_RenderDrawHistogram(rpRenderer, rAxes, 0,0, 0,0,
        xImag,nCoefficients, 0,false/*fill*/, true/*drawAxis*/, title );

    /******************* plot reconstructed function *******************/
    memset( xOriginal,0, sizeof(fftw_complex)*nCoefficients );
    fftw_execute(planInverse);
    for ( int i=0; i < nCoefficients; ++i )
    {
        xReal[i] = xOriginal[i][0];
        xImag[i] = xOriginal[i][1];
    }

    SDL_RenderDrawArrow( rpRenderer, rAxes.x+rAxes.w*1.2, rAxes.y+rAxes.h/2,
                                     rAxes.x+rAxes.w*1.4, rAxes.y+rAxes.h/2 );
    rAxes.x += 2.0*rAxes.w;
    sprintf(title,"Re IFT[FT[%s]]",rFuncTitle);
    SDL_RenderDrawHistogram(rpRenderer, rAxes, -M_PI,M_PI, 0,0,
        xReal,nCoefficients, 0,false/*fill*/, true/*drawAxis*/, title );

    sprintf(title,"Im IFT[FT[%s]]",rFuncTitle);
    rAxes.x += 1.5*rAxes.w;
    SDL_RenderDrawHistogram(rpRenderer, rAxes, 0,0, 0,0,
        xImag,nCoefficients, 0,false/*fill*/, true/*drawAxis*/, title );

    delete[] xReal;
    delete[] xImag;

    fftw_destroy_plan(planForward);
    fftw_destroy_plan(planInverse);
    fftw_free(xOriginal);
    fftw_free(xTransformed);
}

void testFftw2d(SDL_Renderer * rpRenderer)
{
    SDL_Rect axes = { 40,40, 100,80 };
    testFftw2dAndPrint( rpRenderer,axes, [](float x){return std::sin(1*x);}, "Sin(x)"  ); axes.y += 110;
    testFftw2dAndPrint( rpRenderer,axes, [](float x){return std::sin(2*x);}, "Sin(2x)" ); axes.y += 110;
    testFftw2dAndPrint( rpRenderer,axes, [](float x){return std::sin(3*x);}, "Sin(3x)" ); axes.y += 110;
    testFftw2dAndPrint( rpRenderer,axes, [](float x){return std::abs(x);  }, "|x|"     ); axes.y += 110;
    testFftw2dAndPrint( rpRenderer,axes, [](float x){return x>0?1:0;      }, "Theta(x)" );
}


} // namespace imresh
} // namespace test
