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



namespace imresh
{
namespace libs
{


    /**
     * Applies a kernel, i.e. convolution vector, i.e. weighted sum, to data.
     *
     * Every element @f[ x_i @f] is updated to
     * @f[ x_i^* = \sum_{k=-N_w}^{N_w} w_K x_k @f]
     * here @f[ N_w = \frac{\mathrm{rnWeights}-1}{2} @f]
     * If the kernel reaches an edge, the edge colors is extended beyond the
     * edge. This is done, so that a kernel whose sum is 1, still acts as a
     * kind of mean, else the colors to the edge would darken, e.g. when
     * setting those parts of the sum to 0.
     *
     * @tparam     T_PREC datatype to use, e.g. int,float,double,...
     * @param[in]  rData vector onto which to apply the kernel
     * @param[in]  rnData number of elements in rData
     * @param[in]  rWeights the kernel, convulation matrix, mask to use
     * @param[in]  rnWeights length of kernel. Must be an odd number!
     * @param[in]  rnThreads the maximum amount of parallelism supported by
     *             this function. Note that if only 8 cores are available, then
     *             values higher 8 won't result in more OpenMP threads.
     * @param[out] rData will hold the result, meaning this routine works
     *             in-place
     *
     * @todo make buffer work if rnData > bufferSize
     * @todo use T_KERNELSIZE to hardcode and unroll the loops, see if gcc
     *       automatically unrolls the loops if templated
     **/
    template<class T_PREC>
    void applyKernel
    (
        T_PREC * const rData,
        unsigned int const rnData,
        const T_PREC * const rWeights,
        unsigned int const rnWeights,
        unsigned int const rnThreads = 128
    );

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
    (
        T_PREC * const rData,
        unsigned int const rnData,
        double const rSigma
    );

    /**
     * Blurs a 2D vector of elements using a gaussian kernel
     *
     * @f[ \forall i\in N_x,j\in N_y: x_{ij} = \sum\limits_{k=-n}^n
     * \sum\limits_{l=-n}^n \frac{1}{2\pi\sigma^2} e^{-\frac{ r^2 }{ 2\sigma^2}}
     * x_{kl} @f] mit @f[ r = \sqrt{ {\Delta x}^2 + {\Delta y}^2 } =
     * \sqrt{ k^2+l^2 } \Rightarrow x_{ij} = \sum\limits_{k=-n}^n
     *   \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{ k^2 }{ 2\sigma^2} }
     * \sum\limits_{l=-n}^n
     *   \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{ l^2 }{ 2\sigma^2} }
     * @f] With this we have decomposed the 2D convolution in two consequent 1D
     * convolutions! This makes the calculation of the kernel easier.
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
    (
        T_PREC * const rData,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        double const rSigma
    );


    template<class T_PREC>
    void gaussianBlurHorizontal
    (
        T_PREC * const rData,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        double const rSigma
    );


    template<class T_PREC>
    void gaussianBlurVertical
    (
        T_PREC * const rData,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        double const rSigma
    );

    template<class T_PREC>
    void gaussianBlurVerticalUncached
    (
        T_PREC * const rData,
        unsigned int const rnDataX,
        unsigned int const rnDataY,
        double const rSigma
    );


} // namespace libs
} // namespace imresh
