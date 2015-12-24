#pragma once

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


namespace imresh {
namespace test {


void testGaussianBlurVector
( SDL_Renderer * const rpRenderer, SDL_Rect rect,
  float * const data, const unsigned nData,
  const float sigma, const char * const title );

void testGaussian( SDL_Renderer * const rpRenderer );


} // namespace imresh
} // namespace test
