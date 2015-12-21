#pragma once

#include <SDL.h>
#include <cstdlib> // srand, rand, RAND_MAX, malloc, free
#include <cassert>
#include <cstring> // memcpy
#include <cstdio>  // sprintf
#include <cmath>
#include <cfloat>  // FLT_EPSILON
#include "sdl/sdlplot.h"
#include "math/image/gaussian.h"
#include "math/image/cudaGaussian.h"
#include <cuda_runtime_api.h>  // cudaMalloc, cudaFree, ... (yes not cuda.h!)
#include "math/vector/vectorReduce.h"

#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif


using namespace sdlcommon;
using namespace imresh::math::image;


namespace imresh {
namespace test {


/**
 * Plots original, horizontally and vertically blurred intermediary steps
 *
 * Also compares the result of the CPU blur with the CUDA blur
 **/
void testGaussianBlur2d
( SDL_Renderer * const rpRenderer, SDL_Rect rect, float * const data,
  const unsigned nDataX, const unsigned nDataY, const float sigma,
  const char * const title );

void testGaussian2d( SDL_Renderer * const rpRenderer );


} // namespace imresh
} // namespace test
