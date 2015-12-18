#pragma once

#include <iostream>
#include <cmath>
#include <complex>
#include <cstring>  // memcpy

#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif

namespace imresh {
namespace dft {


/**
 * Calculates the discrete Fourier transform
 *
 * @f[ \tilde{x}_k = \sum\limits{n=0}^{N-1} x_n \left[ \cos \left( 
       -2 \pi k \frac{n}{N} \right) + j \sin\left( -2 \pi k \frac{n}{N}
       \right) \right] ,\quad n\in\mathbb{Z} @f]
 *
 * @param[in]  rData vector of data to transform
 * @param[in]  rnData vector length
 * @param[in]  rForward if true, then calculate @f[ \tilde{x}_k @f], else
 *             @f[ x_n @f]
 * @param[out] rData will hold the transformed data (in-place)
 **/
template<class T_PREC>
void dft ( T_PREC * rData, const unsigned rnData, const bool rForward );


} // namespace dft
} // namespace imresh


#include "dft.cpp"
