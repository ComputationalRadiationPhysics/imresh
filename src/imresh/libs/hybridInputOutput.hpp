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

#include <cstdint>      // uint8_t
#include <vector>
#include <climits>      // UINT_MAX
#include <cstddef>      // NULL


namespace imresh
{
namespace libs
{


    template< class T_COMPLEX, class T_MASK_ELEMENT >
    float calculateHioError
    (
        T_COMPLEX      const * const __restrict__ gPrime,
        T_MASK_ELEMENT const * const __restrict__ rIsMasked,
        unsigned int const nElements,
        bool const rInvertMask = false,
        float * const __restrict__ rpTotalError    = NULL,
        float * const __restrict__ rpnMaskedPixels = NULL
    );

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
     * @param[in]  rSize width, height, depth, ... of rIoData and rMask
     * @param[in]  rnCycles maximum cycles to calculate. Use with care,
     *             because the returned 'solution' may not have converged
     *             enough!
     * @param[in]  rTargetError if the absolute maximum of the masked values
     *             is smaller than this, then the masked value will be treated
     *             as zero meaning the algorithm is finished.
     * @param[in]  rBeta this is a parameter for the hybrid input output
     *             algorithm. Note that if 0, then the algorithm can't proceed
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
        uint8_t const * const rMask,
        std::vector<unsigned int> const rSize,
        unsigned int rnCycles = UINT_MAX,
        float rTargetError = 1e-6,
        float rBeta = 0.9,
        unsigned int rnCores = 0
    );


} // namespace libs
} // namespace imresh
