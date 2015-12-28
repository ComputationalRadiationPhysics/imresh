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
#include <omp.h>      // omp_get_num_procs, omp_set_num_procs
#include <fftw3.h>
#include "hybridInputOutput.h"


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
     * @param[in]  rMask Must be of the same size as rIoData. 0 elements denote
     *             that the object isn't here. 1 denotes the object. The
     *             solution returned will have the property that:
     *                 rIoData * rMask == rIoData
     *             except for rounding errors or if the algorithm did not
     *             converge sufficiently.
     * @param[in]  rnHioCycles maximum cycles to calculate. Use with care,
     *             because the returned 'solution' may not have converged
     *             enough!
     * @param[in]  rnCores Number of Cores to utilize in parallel.
     *             If 0, then the value returned by omp_get_num_procs will
     *             be used.
     * @param[out] rIoData will hold the reconstructed object. Currently
     *             only positive real valued objects are supported.
     * @return 0 on success, else error or warning codes.
     **/
    int hybridInputOutput
    (
        float * const rIoData,
        const uint8_t * const rMask,
        const unsigned rNx,
        const unsigned rNy,
        int rnCycles = INT_MAX,
        float rBeta = 0.9,
        unsigned rnCores = 0
    );

}


} // namespace phasereconstruction
} // namespace algorithms
} // namespace imresh