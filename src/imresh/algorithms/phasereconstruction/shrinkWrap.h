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

#include <cstddef>    // NULL
#include <cstdint>    // uint8_t
#include <climits>    // INT_MAX
#include <cstring>    // memcpy
#include <cassert>
#include <cfloat>     // FLT_EPSILON
#include <cmath>
#include <iostream>
#include <vector>
#include <omp.h>      // omp_get_num_procs, omp_set_num_procs
#include <fftw3.h>
#include "shrinkWrap.h"
#include "algorithms/gaussian.h"
#include "algorithms/phasereconstruction/hybridInputOutput.h"
#include "algorithms/vectorReduce.h"
#include "algorithms/vectorElementwise.h"


namespace imresh
{
namespace algorithms
{
namespace phasereconstruction
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
    int shrinkWrap
    (
        float * const & rIoData,
        const std::vector<unsigned> & rSize,
        unsigned rnHioCycles = 20,
        float rTargetError = 1e-5,
        float rHioBeta = 0.9,
        float rIntensityCutOffAutoCorel = 0.04,
        float rIntensityCutOff = 0.20,
        float sigma0 = 3.0,
        float rSigmaChange = 0.01,
        unsigned rnCycles = 10,
        unsigned rnCores = 1
    );


} // namespace phasereconstruction
} // namespace algorithms
} // namespace imresh
