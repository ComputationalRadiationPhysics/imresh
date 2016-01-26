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

#include "cudaVectorElementwise.hpp"

#include <cassert>
#include <cmath>        // sqrtf
#include <cuda.h>       // atomicCAS
#include <cufft.h>      // cufftComplex, cufftDoubleComplex
#include "libs/cudacommon.h"


namespace imresh
{
namespace algorithms
{
namespace cuda
{


    template< class T_COMPLEX, class T_PREC >
    __global__ void cudaKernelCopyToRealPart
    (
        T_COMPLEX * const rTargetComplexArray,
        T_PREC    * const rSourceRealArray,
        unsigned    const rnElements
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int nTotalThreads = gridDim.x * blockDim.x;
        for ( ; i < rnElements; i += nTotalThreads )
        {
            rTargetComplexArray[i].x = rSourceRealArray[i]; /* Re */
            rTargetComplexArray[i].y = 0;
        }
    }


    template< class T_PREC, class T_COMPLEX >
    __global__ void cudaKernelCopyFromRealPart
    (
        T_PREC    * const rTargetComplexArray,
        T_COMPLEX * const rSourceRealArray,
        unsigned    const rnElements
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int nTotalThreads = gridDim.x * blockDim.x;
        for ( ; i < rnElements; i += nTotalThreads )
        {
            rTargetComplexArray[i] = rSourceRealArray[i].x; /* Re */
        }
    }


    template< class T_PREC, class T_COMPLEX >
    __global__ void cudaKernelComplexNormElementwise
    (
        T_PREC * const rdpDataTarget,
        const T_COMPLEX * const rdpDataSource,
        const unsigned rnElements
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int nTotalThreads = gridDim.x * blockDim.x;
        for ( ; i < rnElements; i += nTotalThreads )
        {
            const float & re = rdpDataSource[i].x;
            const float & im = rdpDataSource[i].y;
            rdpDataTarget[i] = sqrtf( re*re + im*im );
        }
    }

    template<class T_COMPLEX>
    __global__ void cudaKernelComplexNormElementwise
    (
        T_COMPLEX * const rdpDataTarget,
        const T_COMPLEX * const rdpDataSource,
        const unsigned rnElements
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int nTotalThreads = gridDim.x * blockDim.x;
        for ( ; i < rnElements; i += nTotalThreads )
        {
            const float & re = rdpDataSource[i].x;
            const float & im = rdpDataSource[i].y;
            rdpDataTarget[i].x = sqrtf( re*re + im*im );
            rdpDataTarget[i].y = 0;
        }
    }


    template< class T_COMPLEX, class T_PREC >
    __global__ void cudaKernelApplyComplexModulus
    (
        T_COMPLEX * const rdpDataTarget,
        const T_COMPLEX * const rdpDataSource,
        const T_PREC * const rdpComplexModulus,
        const unsigned rnElements
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int nTotalThreads = gridDim.x * blockDim.x;
        for ( ; i < rnElements; i += nTotalThreads )
        {
            const auto & re = rdpDataSource[i].x;
            const auto & im = rdpDataSource[i].y;
            auto norm = sqrtf(re*re+im*im);
            if ( norm == 0 ) // in order to avoid NaN
                norm = 1;
            const float factor = rdpComplexModulus[i] / norm;
            rdpDataTarget[i].x = re * factor;
            rdpDataTarget[i].y = im * factor;
        }
    }


    template< class T_PREC >
    __global__ void cudaKernelCutOff
    (
        T_PREC * const rData,
        unsigned const rnElements,
        const T_PREC rThreshold,
        const T_PREC rLowerValue,
        const T_PREC rUpperValue
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int nTotalThreads = gridDim.x * blockDim.x;
        for ( ; i < rnElements; i += nTotalThreads )
        {
            rData[i] = rData[i] < rThreshold ? rLowerValue : rUpperValue;
        }
    }


    /* kernel call wrappers */


    template< class T_PREC, class T_COMPLEX >
    void cudaComplexNormElementwise
    (
        T_PREC * const rdpDataTarget,
        const T_COMPLEX * const rdpDataSource,
        const unsigned rnElements,
        const cudaStream_t rStream,
        const bool rAsync
    )
    {
        /* 1 operation per thread is a bit low, that's why we let each
         * thread work on 32 elements. Of course for smaller iamges this
         * could mean that the GPU is not fully utilized. But in that
         * case 1 vs. 32 times doesn't make much of a difference anyways */
        const int nThreads = rnElements / 32;
        const int nBlocks = ( nThreads + 256-1 ) / 256;
        cudaKernelComplexNormElementwise<<< nBlocks, 256, 0, rStream >>>
            ( rdpDataTarget, rdpDataSource, rnElements );

        if ( not rAsync )
            CUDA_ERROR( cudaStreamSynchronize( rStream ) );
    }


    /* explicit instantiations */

    template
    __global__ void cudaKernelCopyToRealPart<cufftComplex,float>
    (
        cufftComplex * const rTargetComplexArray,
        float * const rSourceRealArray,
        unsigned const rnElements
    );


    template
    __global__ void cudaKernelCopyFromRealPart<float,cufftComplex>
    (
        float * const rTargetComplexArray,
        cufftComplex * const rSourceRealArray,
        unsigned const rnElements
    );


    template
    __global__ void cudaKernelComplexNormElementwise<float,cufftComplex>
    (
        float * const rdpDataTarget,
        const cufftComplex * const rdpDataSource,
        const unsigned rnElements
    );
    template
    void cudaComplexNormElementwise<float, cufftComplex>
    (
        float * const rdpDataTarget,
        const cufftComplex * const rdpDataSource,
        const unsigned rnElements,
        const cudaStream_t rStream,
        const bool rAsync
    );
    template
    void cudaComplexNormElementwise<cufftComplex, cufftComplex>
    (
        cufftComplex * const rdpDataTarget,
        const cufftComplex * const rdpDataSource,
        const unsigned rnElements,
        const cudaStream_t rStream,
        const bool rAsync
    );


    template
    __global__ void cudaKernelApplyComplexModulus<cufftComplex,float>
    (
        cufftComplex * const rdpDataTarget,
        const cufftComplex * const rdpDataSource,
        const float * const rdpComplexModulus,
        const unsigned rnElements
    );


    template
    __global__ void cudaKernelCutOff<float>
    (
        float * const rData,
        unsigned const rnElements,
        const float rThreshold,
        const float rLowerValue,
        const float rUpperValue
    );


} // namespace cuda
} // namespace algorithms
} // namespace imresh
