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


#pragma once

#include <SDL.h>
#include <cstdlib> // srand, rand, RAND_MAX, malloc, free
#include <cassert>
#include <cstring> // memcpy
#include <cstdio>  // sprintf
#include <cmath>
#include <cfloat>  // FLT_EPSILON
#include "sdlcommon/sdlplot.h"
#include "algorithms/gaussian.h"
#include "algorithms/cuda/cudaGaussian.h"
#include <cuda_runtime_api.h>  // cudaMalloc, cudaFree, ... (yes not cuda.h!)
//#include "algorithms/vectorReduce.h"

#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif


namespace imresh
{
namespace test
{


    /**
     * Plots original, horizontally and vertically blurred intermediary steps
     *
     * Also compares the result of the CPU blur with the CUDA blur
     **/
    void testGaussianBlur2d
    (
        SDL_Renderer * const & rpRenderer,
        SDL_Rect rRect,
        float * const & rData,
        const unsigned & rnDataX,
        const unsigned & rnDataY,
        const float & rSigma,
        const char * const & rTitle
    );

    void testGaussian2d
    (
        SDL_Renderer * const & rpRenderer
    );


} // namespace imresh
} // namespace test
