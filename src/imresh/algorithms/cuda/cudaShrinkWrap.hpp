/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2016 Maximilian Knespel, Phillip Trommler
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

#include <cuda_runtime_api.h> // cudaStream_t


namespace imresh
{
namespace algorithms
{
namespace cuda
{


    template< class T_COMPLEX, class T_PREC >
    __global__ void cudaKernelApplyHioDomainConstraints
    (
        T_COMPLEX       * const __restrict__ rdpgPrevious,
        T_COMPLEX const * const __restrict__ rdpgPrime,
        T_PREC    const * const __restrict__ rdpIsMasked,
        unsigned int const rnElements,
        T_PREC const rHioBeta
    );


#   ifdef IMRESH_DEBUG
        /**
         * checks if the imaginary parts are all 0. For debugging purposes
         **/
        template< class T_COMPLEX >
        void checkIfReal
        (
            T_COMPLEX const * const rData,
            unsigned int const rnElements
        );
#   endif


    template< class T_PREC >
    float compareCpuWithGpuArray
    (
        T_PREC const * const __restrict__ rpData,
        T_PREC const * const __restrict__ rdpData,
        unsigned int const rnElements
    );


    /**
     * Finds f(x) so that FourierTransform[f(x)] == Input(x)
     *
     * For all the default parameters you can use -1 to denote that the
     * default value should be used.
     *
     * @param[in]  rIoData measured (phaseless) intensity distribution whose
     *             phase shrinkWrap will reconstruct
     * @param[out] rIoData will hold the reconstructed object. Currently
     *             only positive real valued objects are supported.
     * @param[in] rnBlocks CUDA blocks to use. For small problems but many
     *            images a low block size may be useful.
     * @param[in] rnThreads CUDA threads to use. Note that on
     *            compute capability 3.x the maximum number of concurrent
     *            threads is 2048 per SMM.
     * @param[in] rStream CUDA stream to use
     * @param[in] rnCycles number of shrink-wrap cycles. One shrink-wrap cycle
     *            includes blurring, mask creation and a call to HIO
     * @param[in] rTargetError if the absolute maximum of the masked values
     *            is smaller than this, then the masked value will be treated
     *            as zero meaning the algorithm is finished.
     * @param[in] rHioBeta this is a parameter for the hybrid input output
     *            algorithm. Note that if 0, then the algorithm can't proceed
     * @param[in] rIntensityCutOffAutoCorel In the first step the shrink-wrap
     *            algorithm uses this value to make a mask for the
     *            HIO-algorithm. This is because the initial mask shouldn't be
     *            too strong or else the everything will be masked.
     *            The value is relative to the maximum value. 0 means
     *            everything will be masked, 1.0 means nothing will be masked.
     *            A value of 0 will result in the algorithm doing nothing.
     * @param[in] rIntensityCutOff the cut-off threshold for subsequent steps,
     *            @see rIntensityCutOffAutoCorel
     * @param[in] rSigma0 the first sigma for the gaussian blurs.
     * @param[in] rSigmaChange in the successive shrink-wrap cycles the
     *            partially reconstructed images will be less and less blurred
     *            for the mask creation, meaning sigma is decreased by
     *            rSigmaChange * currentSigma.
     * @param[in] rnHioCycles Maximum number of HIO cycles. This is a safety
     *            net to prevent hangs if the algorithm doesn't progress
     *
     * @return 0 on success, else error or warning codes.
     **/
    int cudaShrinkWrap
    (
        float * const       rIoData,
        unsigned int const  rImageWidth,
        unsigned int const  rImageHeight,
        cudaStream_t const  rStream                     = 0,
        unsigned int        rnBlocks                    = 256,
        unsigned int        rnThreads                   = 256,
        unsigned int        rnCycles                    = 0,
        float               rTargetError                = 0,
        float               rHioBeta                    = 0,
        float               rIntensityCutOffAutoCorel   = 0,
        float               rIntensityCutOff            = 0,
        float               rSigma0                     = 0,
        float               rSigmaChange                = 0,
        unsigned int        rnHioCycles                 = 0
    );


} // namespace cuda
} // namespace algorithms
} // namespace imresh
