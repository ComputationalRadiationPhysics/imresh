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

#include "vectorElementwise.hpp"

#include <cassert>
#include <cmath>        // sqrtf
//#include <cuda.h>       // atomicCAS
#include <cuda_to_cupla.hpp>
//#include <cufft.h>      // cufftComplex, cufftDoubleComplex
//#include "libs/cudacommon.h"


namespace imresh
{
namespace algorithms
{

    int dummy(void) { return 0; };

    template< class T_COMPLEX, class T_PREC >
    template< class T_ACC >
    ALPAKA_FN_ACC void
    cudaKernelApplyHioDomainConstraints<T_COMPLEX, T_PREC>::template operator()(
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

        int i = blockIdx.x * blockDim.x + threadIdx.x;
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

/* TODO

    #define INSTANTIATE_cudaKernelComplexNormElementwise( T_PREC, T_COMPLEX ) \
    template                                                                  \
    __global__ void cudaKernelComplexNormElementwise<T_PREC,T_COMPLEX>        \
    (                                                                         \
        T_PREC * const rdpDataTarget,                                         \
        T_COMPLEX const * const rdpDataSource,                                \
        unsigned int const rnElements                                         \
    );
    INSTANTIATE_cudaKernelComplexNormElementwise( float, cufftComplex )
    #undef INSTANTIATE_cudaKernelComplexNormElementwise
*/

#if false
    template< class T_COMPLEX, class T_PREC >
    __global__ void cudaKernelCopyToRealPart
    (
        T_COMPLEX * const rTargetComplexArray,
        T_PREC    * const rSourceRealArray,
        unsigned int const rnElements
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
        unsigned int const rnElements
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
        T_COMPLEX const * const rdpDataSource,
        unsigned int const rnElements
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
        T_COMPLEX const * const rdpDataSource,
        unsigned int const rnElements
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
        T_COMPLEX const * const rdpDataSource,
        T_PREC const * const rdpComplexModulus,
        unsigned int const rnElements
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
        unsigned int const rnElements,
        T_PREC const rThreshold,
        T_PREC const rLowerValue,
        T_PREC const rUpperValue
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
        T_COMPLEX const * const rdpDataSource,
        unsigned int const rnElements,
        cudaStream_t const rStream,
        bool const rAsync
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
    __global__ void cudaKernelApplyHioDomainConstraints<cufftComplex, float>
    (
        cufftComplex       * const rdpgPrevious,
        cufftComplex const * const rdpgPrime,
        float const * const rdpIsMasked,
        unsigned int const rnElements,
        float const rHioBeta
    );

    template
    __global__ void cudaKernelCopyToRealPart<cufftComplex,float>
    (
        cufftComplex * const rTargetComplexArray,
        float * const rSourceRealArray,
        unsigned int const rnElements
    );


    template
    __global__ void cudaKernelCopyFromRealPart<float,cufftComplex>
    (
        float * const rTargetComplexArray,
        cufftComplex * const rSourceRealArray,
        unsigned int const rnElements
    );


    #define INSTANTIATE_cudaKernelComplexNormElementwise( T_PREC, T_COMPLEX ) \
    template                                                                  \
    __global__ void cudaKernelComplexNormElementwise<T_PREC,T_COMPLEX>        \
    (                                                                         \
        T_PREC * const rdpDataTarget,                                         \
        T_COMPLEX const * const rdpDataSource,                                \
        unsigned int const rnElements                                         \
    );
    INSTANTIATE_cudaKernelComplexNormElementwise( float, cufftComplex )

    #define INSTANTIATE_cudaComplexNormElementwise( T_PREC, T_COMPLEX ) \
    template                                                            \
    void cudaComplexNormElementwise<T_PREC, T_COMPLEX>                  \
    (                                                                   \
        T_PREC * const rdpDataTarget,                                   \
        T_COMPLEX const * const rdpDataSource,                          \
        unsigned int const rnElements,                                  \
        cudaStream_t const rStream,                                     \
        bool const rAsync                                               \
    );
    INSTANTIATE_cudaComplexNormElementwise( float, cufftComplex )
    INSTANTIATE_cudaComplexNormElementwise( cufftComplex, cufftComplex )

    #define INSTANTIATE_cudaKernelApplyComplexModulus( T_COMPLEX, T_PREC )  \
    template                                                                \
    __global__ void cudaKernelApplyComplexModulus<T_COMPLEX,T_PREC>         \
    (                                                                       \
        T_COMPLEX * const rdpDataTarget,                                    \
        T_COMPLEX const * const rdpDataSource,                              \
        T_PREC const * const rdpComplexModulus,                             \
        unsigned int const rnElements                                       \
    );
    INSTANTIATE_cudaKernelApplyComplexModulus( cufftComplex, float )

    #define INSTANTIATE_cudaKernelCutOff( T_PREC )  \
    template                                        \
    __global__ void cudaKernelCutOff<T_PREC>        \
    (                                               \
        T_PREC * const rData,                       \
        unsigned int const rnElements,              \
        T_PREC const rThreshold,                    \
        T_PREC const rLowerValue,                   \
        T_PREC const rUpperValue                    \
    );
    INSTANTIATE_cudaKernelCutOff( float )
    INSTANTIATE_cudaKernelCutOff( double )

#endif

} // namespace algorithms
} // namespace imresh
