#pragma once

#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>  // memcpy
#include <cstddef>  // NULL

/**
 * Applies a kernel, i.e. convolution vector, i.e. weighted sum, to data.
 *
 * Every element @f[ x_i @f] is updated to
 * @f[ x_i^* = \sum_{k=-N_w}^{N_w} w_K x_k @f]
 * here @f[ N_w = \frac{\mathrm{rnWeights}-1}{2} @f]
 * If the kernel reaches an edge, the edge colors is extended beyond the edge.
 * This is done, so that a kernel whose sum is 1, still acts as a kind of mean,
 * else the colors to the edge would darken, e.g. when setting those parts of
 * the sum to 0.
 *
 * @tparam     T_PREC datatype to use, e.g. int,float,double,...
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

/**
 * Calculates the weights for a gaussian kernel
 *
 * @param[in]  T_PREC precision. Should only be a floating point type. For
 *             integers the sum of the weights may not be 1!
 * @param[in]  rSigma standard deviation for the gaussian. This determines the
 *             kernel size
 * @param[out] rWeights array the kernel will be written into
 * @param[in]  rnWeights maximum writable size of rWeights
 * @return kernel size. If the returned kernel size > rnWeights, then rWeights
 *         wasn't changed. Normally you would want to check for that, allocate
 *         a larger array and call this function again.
 **/
template<class T_PREC>
int calcGaussianKernel
( double rSigma, T_PREC * rWeights, const int rnWeights );

/**
 * Blurs a 1D vector of elements using a gaussian kernel
 *
 * @param[in]  rData vector to blur
 * @param[in]  rnData length of rData
 * @param[in]  rSigma standard deviation of gaussian to use. Higher means
 *             a blurrier result.
 * @param[out] rData blurred vector (in-place)
 **/
template<class T_PREC>
void gaussianBlur
( T_PREC * rData, int rnData, double rSigma );

/**
 * Blurs a 1D vector of elements using a gaussian kernel
 *
 *
 *
 * @param[in]  rData vector to blur
 * @param[in]  rnDataX number of columns in matrix, i.e. line length
 * @param[in]  rnDataY number of rows in matrix, i.e. number of lines
 * @param[in]  rSigma standard deviation of gaussian to use. Higher means
 *             a blurrier result.
 * @param[out] rData blurred vector (in-place)
 **/
template<class T_PREC>
void gaussianBlur
( T_PREC * rData, int rnDataX, int rnDataY, double rSigma );

template<class T_PREC>
void gaussianBlurHorizontal
( T_PREC * rData, int rnDataX, int rnDataY, double rSigma );

template<class T_PREC>
void gaussianBlurVertical
( T_PREC * rData, int rnDataX, int rnDataY, double rSigma );
