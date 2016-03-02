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

#include <cuda_to_cupla.hpp>    // cudaStream_t


namespace imresh
{
namespace algorithms
{
namespace cuda
{


    /**
     * Blurs a 2D vector of elements using a gaussian kernel
     *
     * @f[ \forall i\in N_x,j\in N_y: x_{ij} = \sum\limits_{k=-n}^n
     * \sum\limits_{l=-n}^n \frac{1}{2\pi\sigma^2} e^{-\frac{ r^2 }{ 2\sigma^2} }
     * x_{kl} @f] mit @f[ r = \sqrt{ {\Delta x}^2 + {\Delta y}^2 } =
     * \sqrt{ k^2+l^2 } \Rightarrow x_{ij} = \sum\limits_{k=-n}^n
     *   \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{ k^2 }{ 2\sigma^2} }
     * \sum\limits_{l=-n}^n
     *   \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{ l^2 }{ 2\sigma^2} }
     * @f] With this we have decomposed the 2D convolution in two consequent 1D
     * convolutions! This makes the calculation of the kernel easier.
     *
     * @param[in]  rdpData vector in GPU memory to blur
     * @param[in]  rnDataX number of columns in matrix, i.e. line length
     * @param[in]  rnDataY number of rows in matrix, i.e. number of lines
     * @param[in]  rSigma standard deviation of gaussian to use. Higher means
     *             a blurrier result.
     * @param[in]  rStream CUDA stream to use
     * @param[in]  rAsync if true, then don't wait for the CUDA kernel to
     *             finish, else call cudaStreamSynchronize on rStream.
     * @param[out] rdpData blurred vector (in-place)
     **/
    template<class T_PREC>
    void cudaGaussianBlur
    (
        T_PREC * rdpData,
        unsigned int rnDataX,
        unsigned int rnDataY,
        double rSigma,
        cudaStream_t rStream = 0,
        bool rAsync = false
    );


    template<class T_PREC>
    void cudaGaussianBlurVertical
    (
        T_PREC * rdpData,
        unsigned int rnDataX,
        unsigned int rnDataY,
        double rSigma,
        cudaStream_t rStream = 0,
        bool rAsync = false
    );


    /**
     * Should only be used for benchmarking purposes.
     *
     * Buffers the kernel to shared memory.
     * @see cudaGaussianBlurHorizontal
     **/
    template<class T_PREC>
    void cudaGaussianBlurHorizontalSharedWeights
    (
        T_PREC * rdpData,
        unsigned int rnDataX,
        unsigned int rnDataY,
        double rSigma,
        cudaStream_t rStream = 0,
        bool rAsync = false
    );


    /**
     * Apply Gaussian blur on a 2D data set in the horizontal axis.
     *
     * This function is just a wrapper which should call one of the
     * 'cudaGaussianBlurHorizontal.+' (regex) functions above. Normally
     * it should be the fastest version.
     *
     * 'Horizontal' means lines of contiguous memory.
     *
     * @see cudaGaussianBlur
     **/
    template<class T_PREC>
    inline void cudaGaussianBlurHorizontal
    (
        T_PREC * rdpData,
        unsigned int rnDataX,
        unsigned int rnDataY,
        double rSigma,
        cudaStream_t rStream = 0,
        bool rAsync = false
    )
    {
        cudaGaussianBlurHorizontalSharedWeights
        ( rdpData, rnDataX, rnDataY, rSigma, rStream, rAsync );
    }


} // namespace cuda
} // namespace algorithms
} // namespace imresh
