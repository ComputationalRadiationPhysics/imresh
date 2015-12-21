
#include <cmath>
#include <SDL.h>    // SDL_Rect
#include "sdl/sdlplot.h"
#include "math/fouriertransform/dft.h"

#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif


namespace imresh {
namespace test {


template<class F>
void testDftAndPrint
( SDL_Renderer * rpRenderer, SDL_Rect rAxes, F f, const char* rFuncTitle )
{
    using namespace sdlcommon;
    using namespace imresh::math::fouriertransform;
    using complex = std::complex<float>;

    const int nCoefficients = 31;
    complex * xOriginal = new complex[nCoefficients];
    float * xReal = new float[nCoefficients];
    float * xImag = new float[nCoefficients];

    for ( int i=0; i < nCoefficients; ++i )
        xOriginal[i].real( f( 2.*M_PI*i/nCoefficients - M_PI ) );

    /******************* plot original function *******************/
    for ( int i=0; i < nCoefficients; ++i )
        xReal[i] = xOriginal[i].real();
    SDL_RenderDrawHistogram( rpRenderer, rAxes, -M_PI,M_PI ,0,0, xReal,
                             nCoefficients, 0,false/*fill*/, true/*drawAxis*/,
                             rFuncTitle );
    SDL_RenderDrawArrow( rpRenderer, rAxes.x+rAxes.w*1.2, rAxes.y+rAxes.h/2,
                                     rAxes.x+rAxes.w*1.4, rAxes.y+rAxes.h/2 );

    /******************* plot coefficients *******************/
    dft( xOriginal,nCoefficients,true );
    for ( int i=0; i < nCoefficients; ++i )
    {
        xReal[i] = xOriginal[i].real();
        xImag[i] = xOriginal[i].imag();
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
    dft( xOriginal,nCoefficients,false /*inverse*/ );
    for ( int i=0; i < nCoefficients; ++i )
    {
        xReal[i] = xOriginal[i].real();
        xImag[i] = xOriginal[i].imag();
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

    delete[] xOriginal;
    delete[] xReal;
    delete[] xImag;
}

void testDft(SDL_Renderer * rpRenderer)
{
    SDL_Rect axes = { 40,40, 100,80 };
    testDftAndPrint( rpRenderer,axes, [](float x){return std::sin(1*x);}, "Sin(x)"  ); axes.y += 110;
    testDftAndPrint( rpRenderer,axes, [](float x){return std::sin(2*x);}, "Sin(2x)" ); axes.y += 110;
    testDftAndPrint( rpRenderer,axes, [](float x){return std::sin(3*x);}, "Sin(3x)" ); axes.y += 110;
    testDftAndPrint( rpRenderer,axes, [](float x){return std::abs(x);  }, "|x|"     ); axes.y += 110;
    testDftAndPrint( rpRenderer,axes, [](float x){return x>0?1:0;      }, "Theta(x)" );
}


} // namespace imresh
} // namespace test
