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
namespace algorithms
{


    struct TestGaussian
    {
        float * pData, * dpData, * pResult, * pResultCpu, * pSolution;
        static constexpr unsigned int nMaxElements = 4*1024*1024;
        static constexpr int maxKernelWidth = 30; // (sigma=4), needed to calculate upper bound of maximum rounding error


        void compareFloatArray
        (
            float * pData,
            float * pResult,
            unsigned int nCols,
            unsigned int nRows,
            float sigma,
            unsigned line = 0
        );

        /**
         * Makes some general checks which should hold true for gaussian blurs
         *
         *  - mean should be unchanged (only for periodic boundary conditions)
         *  - new values shouldn't get larger than largest value of original
         *    data, smallest shouldn't get smaller.
         *  - the sum of absolute differences to neighboring elements should
         *    be smaller or equal than in original data (to proof ..)
         *    -> this also should be true at every point!:
         *       data: {x_i} -> gaussian -> {y_i} with y_i = \sum a_k x_{i+k},
         *       0 \leq a_k \leq 1, \sum a_k = 1, k=-m,...,+m, a_{|k|} \leq a_0
         *       now look at |y_{i+1} - y_i| = |\sum a_k [ x_{i+1+k} - x_{i+k} ]|
         *       \leq \sum a_k | x_{i+1+k} - x_{i+k} |
         *       this means element wise this isn't true, because the evening
         *       of (x_i) is equivalent to an evening of (x_i-x_{i-1}) and this
         *       means e.g. for 1,...,1,7,1,0,..,0 the absolute differences are:
         *       0,...,0,6,6,1,0,...,0 so 13 in total
         *       after evening e.g. with kernel 1,1,0 we get:
         *       1,...,1,4,4,0.5,0,...,0 -> diff: 0,...,0,3,0,2.5,0.5,0,...,0
         *       so in 6 in total < 13 (above), but e.g. the difference to the
         *       right changed from 0 to 0.5 or from1 to 2.5, meaning it
         *       increased, because it was next to a very large difference.
         *    -> proof for the sum: (assuming periodic values, or compact, i.e.
         *       finite support)
         *       totDiff' = \sum_i |y_{i+1} - y_i|
         *             \leq \sum_i \sum_k a_k | x_{i+1+k} - x_{i+k} |
         *                = \sum_k a_k \sum_i | x_{i+1+k} - x_{i+k} |
         *                = \sum_i | x_{i+1} - x_{i} |
         *       the last step is true because of \sum_k a_k = 1 and because
         *       \sum_i goes over all (periodic) values so that we can group
         *       2*m+1 identical values for \sum_k a_k const to be 1.
         *       (I think this could be shown more formally with some kind of
         *       index tricks)
         *       - we may have used periodic conditions for the proof above, but
         *         the extension of the border introduces only 0 differences and
         *         after the extended values can also be seen as periodic formally,
         *         so that the proof also works for this case.
         *
         * @param[in] pResult blurred data
         * @param[in] pOriginal raw unblurred data
         * @param[in] nElements number of elements to check in pResult and
         *            pOriginal (Both must be nStride*nElements elements large, or
         *            else invalid memory accesses will occur)
         * @param[in] nStride if 1, then contiguous elements will be checked.
         *            Can be useful to check columns in 2D data, by setting
         *            nStride = nCols. Must not be 0
         **/
        void checkGaussian
        (
            float const * pResult,
            float const * pOriginal,
            unsigned int nElements,
            unsigned int nStride = 1
        );

        /**
         * Calls checkGaussian for every row
         **/
        void checkGaussianHorizontal
        (
            float const * pResult,
            float const * pOriginal,
            unsigned int nCols,
            unsigned int nRows
        );

        /**
         * Calls checkGaussian for every column
         **/
        void checkGaussianVertical
        (
            float const * pResult,
            float const * pOriginal,
            unsigned int nCols,
            unsigned int nRows
        );

        void checkIfElementsEqual
        (
            float const * pData,
            unsigned int nData,
            unsigned int nStride = 1
        );

        void fillWithRandomValues
        (
            float * dpData,
            float * pData,
            unsigned int nElements
        );

        void testGaussianDiracDeltas( void );
        void testGaussianRandomSingleData( void );
        void testGaussianConstantValuesPerRowLine( void );
        void testGaussianConstantValues( void );
        void benchmarkGaussianGeneralRandomValues( void );
        void operator()( void );

    }; // struct TestGaussian


} // namespace algorithms
} // namespace imresh
