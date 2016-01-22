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


#ifndef NDEBUG
#   define DEBUG_CUDASHRINKWRAP 0  // change this if you want to turn on debugging
#else
#   define DEBUG_CUDASHRINKWRAP 0  // leave this as it is
#endif


#include <cstddef>    // NULL
#include <cstring>    // memcpy
#include <cassert>
#include <cstdint>    // uint64_t
#include <cmath>
#include <vector>
#include <cuda.h>     // atomicCAS
#include <cufft.h>
#include <utility>      // std::pair
#include "algorithms/cuda/cudaGaussian.h"
#if DEBUG_CUDASHRINKWRAP == 1
#    include <fftw3.h>    // kinda problematic to mix this with cufft, but should work if it isn't cufftw.h
#    include "algorithms/vectorReduce.hpp"
#    include "algorithms/vectorElementwise.hpp"
#endif


namespace imresh
{
namespace algorithms
{
namespace cuda
{


    /**
     * Finds f(x) so that FourierTransform[f(x)] == Input(x)
     *
     * For all the default parameters you can use -1 to denote that the
     * default value should be used.
     *
     * @param[in]  rIoData measured (phaseless) intensity distribution whose
     *             phase shrinkWrap will reconstruct
     * @param[in]  rnCores Number of Cores to utilize in parallel.
     *             (If 0 then it tries to choose automatically)
     * @param[out] rIoData will hold the reconstructed object. Currently
     *             only positive real valued objects are supported.
     * @return 0 on success, else error or warning codes.
     **/
    int cudaShrinkWrap
    (
        float * const & rIoData,
        const std::vector<unsigned> & rSize,
        unsigned rnCycles = 20,
        float rTargetError = 1e-5,
        float rHioBeta = 0.9,
        float rIntensityCutOffAutoCorel = 0.04,
        float rIntensityCutOff = 0.20,
        float sigma0 = 3.0,
        float rSigmaChange = 0.01,
        unsigned rnHioCycles = 20,
        unsigned rnCores = 1
    );

    int shrinkWrap
    (
        float* const& rIntensity,
        const std::pair<unsigned,unsigned>& rSize,
        cudaStream_t strm,
        unsigned rnCycles,
        float rTargetError,
        float rHioBeta,
        float rIntensityCutOffAutoCorel,
        float rIntensityCutOff,
        float sigma0,
        float rSigmaChange,
        unsigned rnHioCycles,
        unsigned rnCores
    );
} // namespace cuda
} // namespace algorithms
} // namespace imresh
