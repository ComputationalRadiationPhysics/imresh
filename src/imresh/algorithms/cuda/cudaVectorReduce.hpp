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
namespace cuda
{


    /**
     * Saves result of vector reduce in b
     *
     * e.g. call with kernelVectorReduceShared<<<4,128>>>( data, 1888, 4, result,
     *  [](float a, float b){ return fmax(a,b); } )
     * @todo use recursion in order to implement a log_2(n) algorithm
     *
     * @param[in]  rData      vector to reduce
     * @param[in]  rnData     length of vector to reduce in elements
     * @param[in]  rResult    reduced result value (sum, max, min,..)
     * @param[in]  rInitValue initial value for reduction, e.g. 0 for sum or max
     *                        and FLT_MAX for min
     **/
    template<class T_PREC, class T_FUNC>
    __global__ void kernelVectorReduceShared
    (
        const T_PREC * const rdpData,
        const unsigned rnData,
        T_PREC * const rdpResult,
        T_FUNC f,
        const T_PREC rInitValue
    );


    template<class T_PREC>
    T_PREC cudaVectorMin
    (
        const T_PREC * const rdpData,
        const unsigned rnElements
    );


    template<class T_PREC>
    T_PREC cudaVectorMax
    (
        const T_PREC * const rdpData,
        const unsigned rnElements
    );


    template<class T_PREC>
    T_PREC cudaVectorSum
    (
        const T_PREC * const rdpData,
        const unsigned rnElements
    );


    template< class T_COMPLEX, class T_MASK_ELEMENT >
    __global__ void cudaKernelCalculateHioError
    (
        const T_COMPLEX * const rdpgPrime,
        const T_MASK_ELEMENT * const rdpIsMasked,
        const unsigned rnData,
        const bool rInvertMask,
        float * const rdpTotalError,
        float * const rdpnMaskedPixels
    );


    template<class T_COMPLEX, class T_MASK_ELEMENT>
    float calculateHioError
    (
        const T_COMPLEX * const & rdpData,
        const T_MASK_ELEMENT * const & rdpIsMasked,
        const unsigned & rnElements,
        const bool & rInvertMask = false
    );


    template<class T_PREC>
    T_PREC cudaVectorMaxSharedMemory
    (
        const T_PREC * const rdpData,
        const unsigned rnElements
    );

    template<class T_PREC>
    T_PREC cudaVectorMaxSharedMemoryWarps
    (
        const T_PREC * const rdpData,
        const unsigned rnElements
    );


} // namespace cuda
} // namespace algorithms
} // namespace imresh
