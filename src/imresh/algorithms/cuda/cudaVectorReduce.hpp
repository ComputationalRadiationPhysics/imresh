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

#include <cuda_runtime_api.h>   // cudaStream_t
#include <cmath>                // fmax


namespace imresh
{
namespace algorithms
{
namespace cuda
{

    /**
     * simple functors to just get the sum of two numbers. To be used
     * for the binary vectorReduce function to make it a vectorSum or
     * vectorMin or vectorMax
     **/
    template<class T> struct SumFunctor {
        __device__ __host__ inline T operator() ( const T & a, const T & b )
        { return a+b; }
    };
    template<class T> struct MinFunctor {
        __device__ __host__ inline T operator() ( const T & a, const T & b )
        { if (a<b) return a; else return b; } // std::min not possible, can't call host function from device!
    };
    template<class T> struct MaxFunctor {
        __device__ __host__ inline T operator() ( const T & a, const T & b )
        { if (a>b) return a; else return b; }
    };
    template<> struct MaxFunctor<float> {
        __device__ __host__ inline float operator() ( const float & a, const float & b )
        { return fmax(a,b); }
    };

    /**
     * @see kernelVectorReduceWarps but uses only shared memory to reduce
     * per block
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

    /**
     * @see kernelVectorReduceWarps but uses also shared memory to reduce
     * per block
     **/
    template<class T_PREC, class T_FUNC>
    __global__ void kernelVectorReduceSharedWarps
    (
        const T_PREC * const rdpData,
        const unsigned rnData,
        T_PREC * const rdpResult,
        T_FUNC f,
        const T_PREC rInitValue
    );

    /**
     * Uses __shfl to reduce per warp before atomicAdd to global memory
     *
     * e.g. call with kernelVectorReduceShared<<<4,128>>>(
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
    __global__ void kernelVectorReduceWarps
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
        const unsigned rnElements,
        cudaStream_t rStream = 0
    );


    template<class T_PREC>
    T_PREC cudaVectorMax
    (
        const T_PREC * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream = 0
    );


    template<class T_PREC>
    T_PREC cudaVectorSum
    (
        const T_PREC * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream = 0
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
        const bool & rInvertMask = false,
        cudaStream_t rStream = 0
    );


    template<class T_PREC>
    T_PREC cudaVectorMaxSharedMemory
    (
        const T_PREC * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream = 0
    );

    template<class T_PREC>
    T_PREC cudaVectorMaxSharedMemoryWarps
    (
        const T_PREC * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream = 0
    );


} // namespace cuda
} // namespace algorithms
} // namespace imresh
