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

#include "cudaVectorReduce.hpp"

#include <cassert>
#include <cstdio>
#include <cstdint>    // uint64_t
#include <limits>     // lowest
#include <cmath>
#include <cuda.h>     // atomicCAS
#include <cufft.h>    // cufftComplex, cufftDoubleComplex
#include "libs/cudacommon.hpp"


namespace imresh
{
namespace algorithms
{
namespace cuda
{


    SumFunctor<float > sumFunctorf;
    MinFunctor<float > minFunctorf;
    MaxFunctor<float > maxFunctorf;
    SumFunctor<double> sumFunctord;
    MinFunctor<double> minFunctord;
    MaxFunctor<double> maxFunctord;


    template<class T_FUNC>
    __device__ inline void atomicFunc
    (
        float * const rdpTarget,
        float const rValue,
        T_FUNC f
    )
    {
        /* atomicCAS only is defined for int and long long int, thats why we
         * need these roundabout casts */
        uint32_t assumed;
        uint32_t old = * (uint32_t*) rdpTarget;

        /* atomicCAS returns the value with which the current value 'assumed'
         * was compared. If the value changed between reading out to assumed
         * and calculating the reduced value and storing it back, then we
         * need to call this function again. (I hope the GPU has some
         * functionality to prevent synchronized i.e. neverending races ... */
        do
        {
            assumed = old;

            /* If the reduced value doesn't change, then we don't need to hinder
             * other threads with atomicCAS. This additional check may prove a
             * bottleneck, if this is rarely the case, e.g. for sum and no 0s or
             * for max and an ordered list, where the largest is the last
             * element. In tests this more often slowed down the calculation */
            //if ( f( __int_as_float(assumed), rValue ) == assumed )
            //    break;

            /* compare and swap after the value was read with assumend, return
             * old value, if assumed isn't anymore the value at rdpTarget,
             * then we will have to try again to write it */
            old = atomicCAS( (uint32_t*) rdpTarget, assumed,
                __float_as_int( f( __int_as_float(assumed), rValue ) ) );
        }
        while ( assumed != old );
    }

    template<class T_FUNC>
    __device__ inline void atomicFunc
    (
        double * const rdpTarget,
        double const rValue,
        T_FUNC f
    )
    {
        using ull = unsigned long long int;
        ull assumed;
        ull old = * (ull*) rdpTarget;
        do
        {
            assumed = old;
            old = atomicCAS( (ull*) rdpTarget, assumed,
                __double_as_longlong( f( __longlong_as_double(assumed), rValue ) ) );
        }
        while ( assumed != old );
    }


    template<>
    __device__ inline void atomicFunc<int,MaxFunctor<int>>
    (
        int * const rdpTarget,
        int const rValue,
        MaxFunctor<int> f
    )
    {
        atomicMax( rdpTarget, rValue );
    }


    /*
    // seems to work for testVectorReduce, but it shouldn't oO, maybe just good numbers, or because this is only for max, maybe it wouldn't work for min, because the maximum is > 0 ... In the end it isn't faster than atomicCAS and it doesn't even use floatAsOrderdInt yet, which would make use of bitshift, subtraction and logical or, thereby decreasing performance even more: http://stereopsis.com/radix.html
    template<>
    __device__ inline void atomicFunc<float,MaxFunctor<float>>
    (
        float * const rdpTarget,
        const float rValue,
        MaxFunctor<float> f
    )
    {
        atomicMax( (int*)rdpTarget, __float_as_int(rValue) );
    }*/

    template<class T_PREC, class T_FUNC>
    __global__ void kernelVectorReduce
    (
        T_PREC const * const __restrict__ rdpData,
        unsigned int const rnData,
        T_PREC * const __restrict__ rdpResult,
        T_FUNC f,
        T_PREC const rInitValue
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        auto iElem = rdpData + blockIdx.x * blockDim.x + threadIdx.x;
        auto localReduced = T_PREC( rInitValue );
        #pragma unroll
        for ( ; iElem < rdpData + rnData; iElem += gridDim.x * blockDim.x )
            localReduced = f( localReduced, *iElem );

        /* reduce per warp (warpSize == 32 assumed) */
        int constexpr cWarpSize = 32;
        assert( cWarpSize == warpSize );
        #pragma unroll
        for ( int32_t warpDelta = cWarpSize / 2; warpDelta > 0; warpDelta /= 2)
            localReduced = f( localReduced, __shfl_down( localReduced, warpDelta ) );

        if ( threadIdx.x % cWarpSize == 0 )
            atomicFunc( rdpResult, localReduced, f );
    }


    template<class T_PREC, class T_FUNC>
    T_PREC cudaReduce
    (
        T_PREC const * const rdpData,
        unsigned int const rnElements,
        T_FUNC f,
        T_PREC const rInitValue,
        cudaStream_t rStream
    )
    {
        const unsigned nThreads = 128;
        //const unsigned nBlocks  = ceil( (float) rnElements / nThreads );
        //printf( "nThreads = %i, nBlocks = %i\n", nThreads, nBlocks );
        const unsigned nBlocks = 288;
        /* 256*256 = 65536 concurrent threads should fill most modern graphic
         * cards. E.g. GTX 760 can only handle 12288 runnin concurrently,
         * everything else will be run after some threads finished. The
         * number of kernels is only 384, because of oversubscription with
         * warps */
        assert( nBlocks < 65536 );

        T_PREC reducedValue;
        T_PREC * dpReducedValue;
        T_PREC initValue = rInitValue;

        CUDA_ERROR( cudaMalloc( (void**) &dpReducedValue, sizeof(T_PREC) ) );
        CUDA_ERROR( cudaMemcpyAsync( dpReducedValue, &initValue, sizeof(T_PREC),
                                     cudaMemcpyHostToDevice, rStream ) );

        /* memcpy is on the same stream as kernel will be, so no synchronize needed! */
        kernelVectorReduce<<< nBlocks, nThreads, 0, rStream >>>
            ( rdpData, rnElements, dpReducedValue, f, rInitValue );
        CUDA_ERROR( cudaPeekAtLastError() );

        CUDA_ERROR( cudaStreamSynchronize( rStream ) );
        CUDA_ERROR( cudaMemcpyAsync( &reducedValue, dpReducedValue, sizeof(T_PREC),
                                     cudaMemcpyDeviceToHost, rStream ) );
        CUDA_ERROR( cudaStreamSynchronize( rStream) );
        CUDA_ERROR( cudaFree( dpReducedValue ) );

        return reducedValue;
    }

    template<class T_PREC>
    T_PREC cudaVectorMin
    (
        T_PREC const * const rdpData,
        unsigned int const rnElements,
        cudaStream_t rStream
    )
    {
        MinFunctor<T_PREC> minFunctor;
        return cudaReduce( rdpData, rnElements, minFunctor, std::numeric_limits<T_PREC>::max(), rStream );
    }


    template<class T_PREC>
    T_PREC cudaVectorMax
    (
        T_PREC const * const rdpData,
        unsigned int const rnElements,
        cudaStream_t rStream
    )
    {
        MaxFunctor<T_PREC> maxFunctor;
        return cudaReduce( rdpData, rnElements, maxFunctor, std::numeric_limits<T_PREC>::lowest(), rStream );
    }


    template<class T_PREC>
    T_PREC cudaVectorSum
    (
        T_PREC const * const rdpData,
        unsigned int const rnElements,
        cudaStream_t rStream
    )
    {
        SumFunctor<T_PREC> sumFunctor;
        return cudaReduce( rdpData, rnElements, sumFunctor, T_PREC(0), rStream );
    }

    inline __device__ uint32_t getLaneId( void )
    {
        uint32_t id;
        asm("mov.u32 %0, %%laneid;" : "=r"(id));
        return id;
    }

    /**
     * "For the input-output algorithms the error @f[ E_F @f] is
     *  usually meaningless since the input @f[ g_k(X) @f] is no longer
     *  an estimate of the object. Then the meaningful error
     *  is the object-domain error @f[ E_0 @f] given by Eq. (15)."
     *                                      (Fienup82)
     * Eq.15:
     * @f[ E_{0k}^2 = \sum\limits_{x\in\gamma} |g_k'(x)^2|^2 @f]
     * where @f[ \gamma @f] is the domain at which the constraints are
     * not met. So this is the sum over the domain which should
     * be 0.
     *
     * Eq.16:
     * @f[ E_{Fk}^2 = \sum\limits_{u} |G_k(u) - G_k'(u)|^2 / N^2
     *              = \sum_x |g_k(x) - g_k'(x)|^2 @f]
     *
     * Note that all pointers may not overlap with each other!
     * Some possible restrictions on the gridSize and blockSize
     *   - Every thread should at least do SOME work for the overhead to
     *     amortize. I suspect that 32 elements per thread can be a good
     *     value, but if you can fill the GPU with some other task meanwhile
     *     you should go even higher.
     **/
    template< class T_COMPLEX, class T_MASK >
    __global__ void cudaKernelCalculateHioError
    (
        T_COMPLEX const * const __restrict__ rdpData,
        T_MASK    const * const __restrict__ rdpIsMasked,
        unsigned int const rnData,
        bool const rInvertMask,
        float * const __restrict__ rdpTotalError,
        float * const __restrict__ rdpnMaskedPixels
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        auto const nTotalThreads = gridDim.x * blockDim.x;
        auto iElem = rdpData     + blockIdx.x * blockDim.x + threadIdx.x;
        auto iMask = rdpIsMasked + blockIdx.x * blockDim.x + threadIdx.x;

        float localTotalError    = 0;
        float localnMaskedPixels = 0;
        #pragma unroll
        for ( ; iElem < rdpData + rnData; iElem += nTotalThreads, iMask += nTotalThreads )
        {
            auto const re = iElem->x;
            auto const im = iElem->y;

            /* only add up norm where no object should be (rMask == 0) */
            /* note: invert   + masked   -> unmasked  <=> 1 ? 1 -> 0
             *       noinvert + masked   -> masked    <=> 0 ? 1 -> 1
             *       invert   + unmasked -> masked    <=> 1 ? 0 -> 1
             *       noinvert + unmasked -> unmasked  <=> 0 ? 0 -> 0
             *   => ? is xor    => no thread divergence
             */
            #ifndef NDEBUG
                if ( not ( *iMask == 0 or *iMask == 1 ) )
                {
                    printf( "rdpIsMasked[%i] = %u\n", iMask-rdpIsMasked, *iMask );
                    assert( *iMask == 0 or *iMask == 1 );
                }
            #endif
            const bool shouldBeZero = rInvertMask xor (bool) *iMask;
            assert( *iMask >= 0.0 and *iMask <= 1.0 );
            //float shouldBeZero = rInvertMask + ( 1-2*rInvertMask )**iMask;
            /*
            float shouldBeZero = rdpIsMasked[i];
            if ( rInvertMask )
                shouldBeZero = 1 - shouldBeZero;
            */

            localTotalError    += shouldBeZero * sqrtf( re*re+im*im );
            localnMaskedPixels += shouldBeZero;
        }

        /* reduce per warp (warpSize == 32 assumed) */
        int constexpr cWarpSize = 32;
        assert( cWarpSize == warpSize );
        #pragma unroll
        for ( int32_t warpDelta = cWarpSize / 2; warpDelta > 0; warpDelta /= 2 )
        {
            localTotalError    += __shfl_down( localTotalError   , warpDelta );
            localnMaskedPixels += __shfl_down( localnMaskedPixels, warpDelta );
        }

        assert( getLaneId() == threadIdx.x % cWarpSize );
        if ( getLaneId() == 0 )
        {
            atomicAdd( rdpTotalError   , localTotalError    );
            atomicAdd( rdpnMaskedPixels, localnMaskedPixels );
        }
    }

    template<class T_COMPLEX, class T_MASK>
    float cudaCalculateHioError
    (
        T_COMPLEX const * const rdpData,
        T_MASK const * const rdpIsMasked,
        unsigned int const rnElements,
        bool const rInvertMask,
        cudaStream_t rStream,
        float * const rpTotalError,
        float * const rpnMaskedPixels
    )
    {
        const unsigned nThreads = 256;
        //const unsigned nBlocks  = ceil( (float) rnElements / nThreads );
        const unsigned nBlocks  = 256;
        assert( nBlocks < 65536 );

        float     totalError,     nMaskedPixels;
        float * dpTotalError, * dpnMaskedPixels;

        CUDA_ERROR( cudaMalloc( (void**) &dpTotalError   , sizeof(float) ) );
        CUDA_ERROR( cudaMalloc( (void**) &dpnMaskedPixels, sizeof(float) ) );
        CUDA_ERROR( cudaMemsetAsync( dpTotalError   , 0, sizeof(float), rStream ) );
        CUDA_ERROR( cudaMemsetAsync( dpnMaskedPixels, 0, sizeof(float), rStream ) );

        /* memset is on the same stream as kernel will be, so no synchronize needed! */
        cudaKernelCalculateHioError<<< nBlocks, nThreads, 0, rStream >>>
            ( rdpData, rdpIsMasked, rnElements, rInvertMask, dpTotalError, dpnMaskedPixels );
        CUDA_ERROR( cudaPeekAtLastError() );
        CUDA_ERROR( cudaStreamSynchronize( rStream ) );

        CUDA_ERROR( cudaMemcpyAsync( &totalError   , dpTotalError   , sizeof(float), cudaMemcpyDeviceToHost, rStream ) );
        CUDA_ERROR( cudaMemcpyAsync( &nMaskedPixels, dpnMaskedPixels, sizeof(float), cudaMemcpyDeviceToHost, rStream ) );
        CUDA_ERROR( cudaStreamSynchronize( rStream ) );

        CUDA_ERROR( cudaFree( dpTotalError    ) );
        CUDA_ERROR( cudaFree( dpnMaskedPixels ) );

        if ( rpTotalError != NULL )
            *rpTotalError    = totalError;
        if ( rpnMaskedPixels != NULL )
            *rpnMaskedPixels = nMaskedPixels;

        return sqrtf(totalError) / nMaskedPixels;
    }


    /* explicit template instantiations */

    template
    float cudaVectorMin<float>
    (
        float const * rdpData,
        unsigned int rnElements,
        cudaStream_t rStream
    );
    template
    double cudaVectorMin<double>
    (
        double const * rdpData,
        unsigned int rnElements,
        cudaStream_t rStream
    );


    template
    float cudaVectorMax<float>
    (
        float const * rdpData,
        unsigned int rnElements,
        cudaStream_t rStream
    );
    template
    double cudaVectorMax<double>
    (
        double const * rdpData,
        unsigned int rnElements,
        cudaStream_t rStream
    );


    template
    float cudaVectorSum<float>
    (
        float const * rdpData,
        unsigned int rnElements,
        cudaStream_t rStream
    );
    template
    double cudaVectorSum<double>
    (
        double const * rdpData,
        unsigned int rnElements,
        cudaStream_t rStream
    );

    template
    __global__ void cudaKernelCalculateHioError
    <cufftComplex, float>
    (
        cufftComplex const * rdpgPrime,
        float const * rdpIsMasked,
        unsigned int rnData,
        bool rInvertMask,
        float * rdpTotalError,
        float * rdpnMaskedPixels
    );


    template
    float cudaCalculateHioError
    <cufftComplex, float>
    (
        cufftComplex const * rdpData,
        float const * rdpIsMasked,
        unsigned int rnElements,
        bool rInvertMask,
        cudaStream_t rStream,
        float * rdpTotalError,
        float * rdpnMaskedPixels
    );
    template
    float cudaCalculateHioError
    <cufftComplex, bool>
    (
        cufftComplex const * rdpData,
        bool const * rdpIsMasked,
        unsigned int rnElements,
        bool rInvertMask,
        cudaStream_t rStream,
        float * rdpTotalError,
        float * rdpnMaskedPixels
    );
    template
    float cudaCalculateHioError
    <cufftComplex, unsigned char>
    (
        cufftComplex const * rdpData,
        unsigned char const * rdpIsMasked,
        unsigned int rnElements,
        bool rInvertMask,
        cudaStream_t rStream,
        float * rdpTotalError,
        float * rdpnMaskedPixels
    );


} // namespace cuda
} // namespace algorithms
} // namespace imresh
