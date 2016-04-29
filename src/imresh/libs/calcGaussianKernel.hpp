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
     * Calculates the weights for a gaussian kernel
     *
     * If rnWeights is NULL, then this function will do nothing more than
     * return how large the kernel for the given sigma would be.
     *
     * We need to choose a size depending on the sigma, for which the kernel
     * will still be 100% accurate, even though we cut of quite a bit.
     * For 8bit images the channels can only store 0 to 255. The most extreme
     * case, where the farest neighbors could still influence the current pixel
     * would be all the pixels inside the kernel being 0 and all the others
     * being 255:
     * @verbatim
     *      255  -----+       +-------
     *                |       |
     *        0       |_______|
     *                    ^
     *             pixel to calculate
     *                <------->
     *      kernel size = 7, i.e. Nw = 3
     *  (number of neighbors in each directions)
     * @endverbatim
     * The only way the outer can influence the interior would be, if the
     * weighted sum over all was > 0.5. Meaning:
     * @f[
     * 255 \int\limits_{-\infty}^{x_\mathrm{cutoff}} \frac{1}{\sqrt{2\pi}\sigma}
     * e^{-\frac{x^2}{2\sigma^2}} \mathrm{d}x = 255 \frac{1}{2}
     * \mathrm{erfc}\left( -\frac{ x_\mathrm{cutoff} }{ \sqrt{2}\sigma } \right)
     * \overset{!}{=} 0.5
     * \Rightarrow x_\mathrm{cutoff} = -\sqrt{2}\sigma
     *   \mathrm{erfc}^{-1}\left( \frac{1}{2 \cdot 255} \right)
     *   = -2.884402748387961466 \sigma
     * @f]
     * This result means, that for @f[ \sigma=1 @f] the kernel size should
     * be 3 to the left and 3 to the right, meaning 7 weights large.
     * The center pixel, which we want to update goes is in the range [-0.5,0.5]
     * The neighbor pixel in [-1.5,-0.5], then [-2.5,-1.5]. So we are very
     * very close to the 2.88440, but we should nevertheless include the
     * pixel at [-3.5,-2.5] to be correct.
     *
     * @tparam     T_PREC precision. Should only be a floating point type. For
     *             integers the sum of the weights may not be 1!
     * @param[in]  rSigma standard deviation for the gaussian. This determines
     *             the kernel size
     * @param[out] rWeights array the kernel will be written into
     * @param[in]  rnWeights maximum writable size of rWeights
     * @param[in]  rMinAbsoluteError when using integers an absolute error of
     *             0.5/255 should be targeted, so that for the maximum range
     *             the absolute error never is bigger than 0.5
     * @return kernel size. If the returned kernel size > rnWeights, then
     *         rWeights wasn't changed. Normally you would want to check for
     *         that, allocate a larger array and call this function again.
     **/
    template<class T_PREC>
    int calcGaussianKernel
    (
        const double & rSigma,
        T_PREC * const & rWeights,
        const unsigned & rnWeights,
        const double & rMinAbsoluteError = 0.5/255
    );


} // namespace libs
} // namespace imresh
