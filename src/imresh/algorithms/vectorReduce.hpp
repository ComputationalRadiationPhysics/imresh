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
#include <cmath>                // fmax


namespace imresh
{
namespace algorithms
{


    template<class T_ACC, class T_PREC, class T_FUNC>
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY inline
    void atomicFunc
    (
        T_ACC const     & acc,
        T_PREC * const rdpTarget,
        T_PREC const rValue,
        T_FUNC f
    );

    template<class T_ACC, class T_FUNC>
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY inline
    void atomicFunc
    (
        T_ACC const     & acc,
        float * const rdpTarget,
        float const rValue,
        T_FUNC f
    );

    template<class T_ACC, class T_FUNC>
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY inline
    void atomicFunc
    (
        T_ACC const     & acc,
        double * const rdpTarget,
        double const rValue,
        T_FUNC f
    );

    /**
     * simple functors to just get the sum of two numbers. To be used
     * for the binary vectorReduce function to make it a vectorSum or
     * vectorMin or vectorMax
     **/
    template<class T> struct SumFunctor {
        ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY inline T operator() ( T a, T b ) const
        { return a+b; }
    };
    template<class T> struct MinFunctor {
        ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY inline T operator() ( T a, T b ) const
        { if (a<b) return a; else return b; } // std::min not possible, can't call host function from device!
    };
    template<class T> struct MaxFunctor {
        ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY inline T operator() ( T a, T b ) const
        { if (a>b) return a; else return b; }
    };
    template<> struct MaxFunctor<float> {
        ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY inline float operator() ( float a, float b ) const
        { return fmax(a,b); }
    };


    /**
     * Uses __shfl to reduce per warp before atomicAdd to global memory
     *
     * e.g. call with CUPLA_KERNEL( kernelVectorReduceShared )(4,128)(
     *  data, 1888, 4, result, [](float a, float b){ return fmax(a,b); } )
     *
     * @tparam T_FUNC  Only the functors from this headers are instantiated
     *         for this template type. for other functors you need to
     *         include the body instead of this header or you need to add
     *         it to the list of explicit template instantitions
     *         Reasons against std::function:
     *             @see http://stackoverflow.com/questions/14677997/
     * @tparam T_PREC datatype of array to reduce. Only float and double are
     *         explicitly instantiated, but you could add more easily.
     *
     * @param[in]  rdpData device pointer to array of data to reduce
     * @param[in]  rnData length of array to reduce
     * @param[out] rdpResult pointer to global memory variable which will hold
     *             the reduce result
     * @param[in]  f reduce functor which takes two arguments and returns 1.
     * @param[in]  rInitValue The init value for rdpResult. E.g. for a
     *             summation this should be 0.
     **/
    template<class T_PREC, class T_FUNC>
    struct kernelVectorReduce
    {
        template< typename T_ACC >
        ALPAKA_FN_NO_INLINE_ACC
        void operator()
        (
            T_ACC const & acc,
            T_PREC const * const __restrict__ rdpData,
            unsigned int const rnData,
            T_PREC * const __restrict__ rdpResult,
            T_FUNC f,
            T_PREC const rInitValue
        ) const;
    };


    template<class T_PREC>
    T_PREC cudaVectorMin
    (
        T_PREC const * rdpData,
        unsigned int rnElements,
        cudaStream_t rStream = 0
    );


    template<class T_PREC>
    T_PREC cudaVectorMax
    (
        T_PREC const * rdpData,
        unsigned int rnElements,
        cudaStream_t rStream = 0
    );


    template<class T_PREC>
    T_PREC cudaVectorSum
    (
        T_PREC const * rdpData,
        unsigned int rnElements,
        cudaStream_t rStream = 0
    );


    template< class T_COMPLEX, class T_MASK >
    struct cudaKernelCalculateHioError
    {
        template< typename T_ACC >
        ALPAKA_FN_NO_INLINE_ACC
        void operator()
        (
            T_ACC const & acc,
            T_COMPLEX const * const __restrict__ rdpgPrime,
            T_MASK    const * const __restrict__ rdpIsMasked,
            unsigned int const rnData,
            bool const rInvertMask,
            float * const __restrict__ rdpTotalError,
            float * const __restrict__ rdpnMaskedPixels
        ) const;
    };


    template<class T_COMPLEX, class T_MASK>
    float cudaCalculateHioError
    (
        T_COMPLEX const * rdpData,
        T_MASK const * rdpIsMasked,
        unsigned int rnElements,
        bool rInvertMask = false,
        cudaStream_t rStream = 0,
        float * rpTotalError = NULL,
        float * rpnMaskedPixels = NULL
    );


} // namespace algorithms
} // namespace imresh
