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


#include "cudaVectorElementwise.hpp"

#include <cassert>
#include <cmath>                    // sqrtf
#include <cuda_to_cupla.hpp>
#include <cufft_to_cupla.hpp>       // cufftComplex, cufftDoubleComplex
#include "libs/cudacommon.hpp"
#include <cstdint>                  // uint32_t


namespace imresh
{
namespace algorithms
{
namespace cuda
{


    template< class T_COMPLEX, class T_PREC >
    template< class T_ACC >
    __device__
    void cudaKernelApplyHioDomainConstraints<T_COMPLEX, T_PREC>
    ::template operator()
    (
        T_ACC const & acc,
        T_COMPLEX       * const rdpgPrevious,
        T_COMPLEX const * const rdpgPrime,
        T_PREC    const * const rdpIsMasked,
        unsigned int const rnElements,
        T_PREC const rHioBeta
    ) const
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int nTotalThreads = gridDim.x * blockDim.x;
        for ( ; i < rnElements; i += nTotalThreads )
        {
            if ( rdpIsMasked[i] == 1 or /* g' */ rdpgPrime[i].x < 0 )
            {
                rdpgPrevious[i].x -= rHioBeta * rdpgPrime[i].x;
                rdpgPrevious[i].y -= rHioBeta * rdpgPrime[i].y;
            }
            else
            {
                rdpgPrevious[i].x = rdpgPrime[i].x;
                rdpgPrevious[i].y = rdpgPrime[i].y;
            }
        }
    }


    template< class T_COMPLEX, class T_PREC >
    template< class T_ACC >
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY
    void cudaKernelCopyToRealPart<T_COMPLEX, T_PREC>
    ::template operator()
    (
        T_ACC const & acc,
        T_COMPLEX * const rTargetComplexArray,
        T_PREC    * const rSourceRealArray,
        unsigned int const rnElements
    ) const
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int nTotalThreads = gridDim.x * blockDim.x;
        for ( ; i < rnElements; i += nTotalThreads )
        {
            rTargetComplexArray[i].x = rSourceRealArray[i]; /* Re */
            rTargetComplexArray[i].y = 0;
        }
    }


    template< class T_PREC, class T_COMPLEX >
    template< class T_ACC >
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY
    void cudaKernelCopyFromRealPart<T_PREC, T_COMPLEX>
    ::template operator()
    (
        T_ACC const & acc,
        T_PREC    * const rTargetComplexArray,
        T_COMPLEX * const rSourceRealArray,
        unsigned int const rnElements
    ) const
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int nTotalThreads = gridDim.x * blockDim.x;
        for ( ; i < rnElements; i += nTotalThreads )
        {
            rTargetComplexArray[i] = rSourceRealArray[i].x; /* Re */
        }
    }


    template< class T_PREC, class T_COMPLEX >
    template< class T_ACC >
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY
    void cudaKernelComplexNormElementwise<T_PREC, T_COMPLEX>
    ::template operator()
    (
        T_ACC const & acc,
        T_PREC * const rdpDataTarget,
        T_COMPLEX const * const rdpDataSource,
        unsigned int const rnElements
    ) const
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int nTotalThreads = gridDim.x * blockDim.x;
        for ( ; i < rnElements; i += nTotalThreads )
        {
            const float & re = rdpDataSource[i].x;
            const float & im = rdpDataSource[i].y;
            rdpDataTarget[i] = sqrtf( re*re + im*im );
        }
    }

    /* first partial specialisation of class definition is needed, only
     * after that can the operator() of the specialized class be defined! */
    template< class T_COMPLEX >
    struct cudaKernelComplexNormElementwise< T_COMPLEX, T_COMPLEX >
    {
        template< typename T_ACC >
        ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY
        void operator()
        (
            T_ACC const & acc,
            T_COMPLEX * const rdpDataTarget,
            T_COMPLEX const * const rdpDataSource,
            unsigned int const rnElements
        ) const;
    };


    template< class T_COMPLEX >
    template< class T_ACC >
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY
    void cudaKernelComplexNormElementwise<T_COMPLEX,T_COMPLEX>
    ::template operator()
    (
        T_ACC const & acc,
        T_COMPLEX * const rdpDataTarget,
        T_COMPLEX const * const rdpDataSource,
        unsigned int const rnElements
    ) const
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
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
    template< class T_ACC >
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY
    void cudaKernelApplyComplexModulus<T_COMPLEX, T_PREC>
    ::template operator()
    (
        T_ACC const & acc,
        T_COMPLEX * const rdpDataTarget,
        T_COMPLEX const * const rdpDataSource,
        T_PREC const * const rdpComplexModulus,
        unsigned int const rnElements
    ) const
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
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
    template< class T_ACC >
    ALPAKA_FN_NO_INLINE_ACC_CUDA_ONLY
    void cudaKernelCutOff<T_PREC>
    ::template operator()
    (
        T_ACC const & acc,
        T_PREC * const rData,
        unsigned int const rnElements,
        T_PREC const rThreshold,
        T_PREC const rLowerValue,
        T_PREC const rUpperValue
    ) const
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
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
        T_COMPLEX const * const rdpDataSource,
        unsigned int const rnElements,
        cudaStream_t const rStream,
        bool const rAsync
    )
    {
        /* 1 operation per thread is a bit low, that's why we let each
         * thread work on 32 elements. Of course for smaller images this
         * could mean that the GPU is not fully utilized. But in that
         * case 1 vs. 32 times doesn't make much of a difference anyways */
        int const nThreads = rnElements / 32;
        int const nBlocks = ( nThreads + 256-1 ) / 256;
        CUPLA_KERNEL( cudaKernelComplexNormElementwise< T_PREC, T_COMPLEX > )
            ( nBlocks, 256, 0, rStream ) // kernel config
            ( rdpDataTarget, rdpDataSource, rnElements ); // kernel arguments
        CUDA_ERROR( cudaPeekAtLastError() );

        if ( not rAsync )
            CUDA_ERROR( cudaStreamSynchronize( rStream ) );
    }


} // namespace cuda
} // namespace algorithms
} // namespace imresh
