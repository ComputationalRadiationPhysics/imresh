
#include <fftw3.h>
#include <cstdio>  // sprintf
#include <cmath>
#include <complex>
#include "sdl/sdlplot.h"

#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif


namespace imresh {
namespace test {


template<class F>
void testFftwAndPrint
( SDL_Renderer * rpRenderer, SDL_Rect rAxes, F f, const char* rFuncTitle );

void testFftw(SDL_Renderer * rpRenderer);


} // namespace imresh
} // namespace test
