#pragma once

#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>  // memcpy

/**
 * Applies a kernel, i.e. convulation vector, i.e. weighted sum, to data.
 *
 * Every element @f[ x_i @f] is updated to
 * @f[ x_i^* = \sum_{k=-N_w}^{N_w} w_K x_k @f]
 * here @f[ N_w = \frac{\mathrm{rnWeights}-1}{2} @f]
 * If the kernel reaches an edge, the edge colors is extended beyond the edge.
 * This is done, so that a kernel whose sum is 1, still acts as a kind of mean,
 * else the colors to the edge would darken, e.g. when setting those parts of
 * the sum to 0.
 *
 * @param[in]  T_PREC datatype to use, e.g. int,float,double,...
 * @param[in]  rData vector onto which to apply the kernel
 * @param[in]  rnData number of elements in rData
 * @param[in]  rWeights the kernel, convulation matrix, mask to use
 * @param[in]  rnWeights length of kernel. Must be an odd number!
 * @param[out] rData will hold the result, meaning this routine works in-place
 *
 * @todo make buffer work if rnData > bufferSize
 * @todo use T_KERNELSIZE to hardcode and unroll the loops, see if gcc
 *       automatically unrolls the loops if templated
 **/
template<class T_PREC>
void applyKernel
( T_PREC * rData, const int rnData,
  const T_PREC * rWeights, const int rnWeights );

template<class T_PREC>
void gaussianBlur
( T_PREC * rData, int rnData, double rSigma );

