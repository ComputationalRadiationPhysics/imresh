#pragma once

#include <iostream>
#include <cmath>

#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif



struct SinBase {
    int k;
    float operator()(float x) const { return std::sin(k*x); }
};
struct CosBase {
    int k;
    float operator()(float x) const { return std::cos(k*x); }
};


#include "numeric/integrate.h"

template<class F, class G>
float trigScp(const F & f, const G & g);

/**
 * Calculates fourier coefficients a_k and b_k up for k=1 to k=rnCoefficients
 *
 * @param[in]  f continuous function to transform
 * @param[in]  rnCoefficients number of complex(!) coefficients to calculate,
 *             the returned float array will contain 2*rnCoefficients elements!
 * @param[out] rCoefficients pointer to allocated float array which will
 *             rnCoefficients cosine, i.e. real, fourier coefficients, followed
 *             by nCoefficients sine, i.e. imaginary, fourier coefficeints
 **/
template<class F>
void dcft( F f, int rnCoefficients, float * rCoefficients );



#include "sdlplot.h"

template<class F>
void testDcftAndPrint
( SDL_Renderer * rpRenderer, SDL_Rect rAxes, F f, const char* rFuncTitle );

#include <SDL.h>

void testDcft(SDL_Renderer * rpRenderer);

#include "dcft.cpp"
