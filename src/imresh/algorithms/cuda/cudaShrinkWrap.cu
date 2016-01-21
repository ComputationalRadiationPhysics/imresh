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

#include <utility>                              // std::pair

#include "algorithms/cuda/cudaShrinkWrap.h"
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

        uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const uint64_t nTotalThreads = gridDim.x * blockDim.x;
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

        uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const uint64_t nTotalThreads = gridDim.x * blockDim.x;
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

        uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const uint64_t nTotalThreads = gridDim.x * blockDim.x;
        for ( ; i < rnElements; i += nTotalThreads )
        {
            const float & re = rdpDataSource[i].x;
            const float & im = rdpDataSource[i].y;
            rdpDataTarget[i] = sqrtf( re*re + im*im );
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

        uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const uint64_t nTotalThreads = gridDim.x * blockDim.x;
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

        uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const uint64_t nTotalThreads = gridDim.x * blockDim.x;
        for ( ; i < rnElements; i += nTotalThreads )
        {
            rData[i] = rData[i] < rThreshold ? rLowerValue : rUpperValue;
        }
    }


    template< class T_COMPLEX, class T_PREC >
    __global__ void cudaKernelApplyHioDomainConstraints
    (
        T_COMPLEX * const rdpgPrevious,
        const T_COMPLEX * const rdpgPrime,
        const T_PREC * const rdpIsMasked,
        unsigned const rnElements,
        const T_PREC rHioBeta
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const uint64_t nTotalThreads = gridDim.x * blockDim.x;
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


    /**
     * simple functors to just get the sum of two numbers. To be used
     * for the binary vectorReduce function to make it a vectorSum or
     * vectorMin or vectorMax
     **/
    template<class T> struct SumFunctor {
        __device__ __host__ T operator() ( const T & a, const T & b )
        { return a+b; }
    };
    template<class T> struct MinFunctor {
        __device__ __host__ T operator() ( const T & a, const T & b )
        { if (a<b) return a; else return b; }
    };
    template<class T> struct MaxFunctor {
        __device__ __host__ T operator() ( const T & a, const T & b )
        { if (a>b) return a; else return b; }
    };
    SumFunctor<float> sumFunctor;
    MaxFunctor<float> maxFunctor;
    MinFunctor<float> minFunctor;


    template<class T_FUNC>
    __device__ float atomicFunc
    (
        float * const rdpTarget,
        const float rValue,
        T_FUNC f
    )
    {
        /* atomicCAS only is defined for int and long long int, thats why we
         * need these roundabout casts */
        int assumed;
        int old = * (int*) rdpTarget;

        /* atomicCAS returns the value with which the current value 'assumed'
         * was compared. If the value changed between reading out to assumed
         * and calculating the reduced value and storing it back, then we
         * need to call this function again. (I hope the GPU has some
         * functionality to prevent synchronized i.e. neverending races ... */
        do
        {
            assumed = old;
            old = atomicCAS( (int*) rdpTarget, assumed,
                __float_as_int( f( __int_as_float(assumed), rValue ) ) );
        } while ( assumed != old );

        return __int_as_float( old );
    }


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
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        const uint64_t nTotalThreads = gridDim.x * blockDim.x;
        uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        assert( i < nTotalThreads );

        T_PREC localReduced = T_PREC(rInitValue);
        for ( ; i < rnData; i += nTotalThreads )
            localReduced = f( localReduced, rdpData[i] );

        __shared__ T_PREC smReduced;
        /* master thread of every block shall set shared mem variable to 0 */
        __syncthreads();
        if ( threadIdx.x == 0 )
            smReduced = T_PREC(rInitValue);
        __syncthreads();

        //atomicMax( &smReduced, localReduced );
        atomicFunc( &smReduced, localReduced, f );

        __syncthreads();
        if ( threadIdx.x == 0 )
            atomicFunc( rdpResult, smReduced, f );
    }


    template<class T_PREC, class T_FUNC>
    T_PREC cudaReduce
    (
        const T_PREC * const rdpData,
        const unsigned rnElements,
        T_FUNC f,
        const T_PREC rInitValue
    )
    {
        const unsigned nThreads = 256;
        const unsigned nBlocks  = ceil( (float) rnElements / nThreads );

        float reducedValue;
        float * dpReducedValue;
        float initValue = (float) rInitValue;

        CUDA_ERROR( cudaMalloc( (void**) &dpReducedValue, sizeof(float) ) );
        CUDA_ERROR( cudaMemcpy( dpReducedValue, &initValue, sizeof(float), cudaMemcpyHostToDevice ) );

        kernelVectorReduceShared<<<nBlocks,nThreads>>>
            ( rdpData, rnElements, dpReducedValue, f, rInitValue );

        CUDA_ERROR( cudaDeviceSynchronize() );
        CUDA_ERROR( cudaMemcpy( &reducedValue, dpReducedValue, sizeof(float), cudaMemcpyDeviceToHost ) );

        CUDA_ERROR( cudaFree( dpReducedValue ) );

        return reducedValue;
    }


    /**
     * "For the input-output algorithms the error E_F is
     *  usually meaningless since the input g_k(X) is no longer
     *  an estimate of the object. Then the meaningful error
     *  is the object-domain error E_0 given by Eq. (15)."
     *                                      (Fienup82)
     * Eq.15:
     * @f[ E_{0k}^2 = \sum\limits_{x\in\gamma} |g_k'(x)^2|^2 @f]
     * where \gamma is the domain at which the constraints are
     * not met. SO this is the sum over the domain which should
     * be 0.
     *
     * Eq.16:
     * @f[ E_{Fk}^2 = \sum\limits_{u} |G_k(u) - G_k'(u)|^2 / N^2
                    = \sum_x |g_k(x) - g_k'(x)|^2 @f]
     **/
    template< class T_COMPLEX, class T_MASK_ELEMENT >
    __global__ void cudaKernelCalculateHioError
    (
        const T_COMPLEX * const rdpgPrime,
        const T_MASK_ELEMENT * const rdpIsMasked,
        const unsigned rnData,
        const bool rInvertMask,
        float * const rdpTotalError,
        float * const rdpnMaskedPixels
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        const uint64_t nTotalThreads = gridDim.x * blockDim.x;
        uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        assert( i < nTotalThreads );

        float localTotalError    = 0;
        float localnMaskedPixels = 0;
        for ( ; i < rnData; i += nTotalThreads )
        {
            const auto & re = rdpgPrime[i].x;
            const auto & im = rdpgPrime[i].y;

            /* only add up norm where no object should be (rMask == 0) */
            assert( rdpIsMasked[i] >= 0.0 and rdpIsMasked[i] <= 1.0 );
            float shouldBeZero = rdpIsMasked[i];
            if ( rInvertMask )
                shouldBeZero = 1 - shouldBeZero;

            localTotalError    += shouldBeZero * ( re*re+im*im );
            localnMaskedPixels += shouldBeZero;
        }

        __shared__ float smTotalError, smnMaskedPixels;
        /* master thread of every block shall set shared mem variable to 0 */
        __syncthreads();
        if ( threadIdx.x == 0 )
        {
            smTotalError    = 0;
            smnMaskedPixels = 0;
        }
        __syncthreads();

        SumFunctor<float> sum;
        atomicFunc( &smTotalError   , localTotalError   , sum );
        atomicFunc( &smnMaskedPixels, localnMaskedPixels, sum );

        __syncthreads();
        if ( threadIdx.x == 0 )
        {
            atomicFunc( rdpTotalError, smTotalError, sum );
            atomicFunc( rdpnMaskedPixels, smnMaskedPixels, sum );
        }
    }


    template<class T_COMPLEX, class T_MASK_ELEMENT>
    float calculateHioError
    (
        const T_COMPLEX * const & rdpData,
        const T_MASK_ELEMENT * const & rdpIsMasked,
        const unsigned & rnElements,
        const bool & rInvertMask = false
    )
    {
        const unsigned nThreads = 256;
        const unsigned nBlocks  = ceil( (float) rnElements / nThreads );

        float     totalError,     nMaskedPixels;
        float * dpTotalError, * dpnMaskedPixels;

        CUDA_ERROR( cudaMalloc( (void**) &dpTotalError   , sizeof(float) ) );
        CUDA_ERROR( cudaMalloc( (void**) &dpnMaskedPixels, sizeof(float) ) );
        CUDA_ERROR( cudaMemset( dpTotalError   , 0, sizeof(float) ) );
        CUDA_ERROR( cudaMemset( dpnMaskedPixels, 0, sizeof(float) ) );

        cudaKernelCalculateHioError<<<nBlocks,nThreads>>>
            ( rdpData, rdpIsMasked, rnElements, rInvertMask, dpTotalError, dpnMaskedPixels );
        CUDA_ERROR( cudaDeviceSynchronize() );

        CUDA_ERROR( cudaMemcpy( &totalError, dpTotalError, sizeof(float), cudaMemcpyDeviceToHost ) );
        CUDA_ERROR( cudaMemcpy( &nMaskedPixels, dpnMaskedPixels, sizeof(float), cudaMemcpyDeviceToHost ) );

        CUDA_ERROR( cudaFree( dpTotalError    ) );
        CUDA_ERROR( cudaFree( dpnMaskedPixels ) );

        return sqrtf(totalError) / nMaskedPixels;
    }


    /**
     * Shifts the Fourier transform result in frequency space to the center
     *
     * @verbatim
     *        +------------+      +------------+          +------------+
     *        |            |      |78 ++  ++ 56|          |     --     |
     *        |            |      |o> ''  '' <o|          | .. <oo> .. |
     *        |     #      |  FT  |-          -| fftshift | ++ 1234 ++ |
     *        |     #      |  ->  |-          -|  ----->  | ++ 5678 ++ |
     *        |            |      |o> ..  .. <o|          | '' <oo> '' |
     *        |            |      |34 ++  ++ 12|          |     --     |
     *        +------------+      +------------+          +------------+
     *                           k=0         k=N-1              k=0
     * @endverbatim
     * This index shift can be done by a simple shift followed by a modulo:
     *   newArray[i] = array[ (i+N/2)%N ]
     **/
    template< class T_COMPLEX >
    void fftShift
    (
        T_COMPLEX * const & data,
        const unsigned & Nx,
        const unsigned & Ny
    )
    {
        /* only up to Ny/2 necessary, because wie use std::swap meaning we correct
         * two elements with 1 operation */
        for ( unsigned iy = 0; iy < Ny/2; ++iy )
        for ( unsigned ix = 0; ix < Nx; ++ix )
        {
            const unsigned index =
                ( ( iy+Ny/2 ) % Ny ) * Nx +
                ( ( ix+Nx/2 ) % Nx );
            std::swap( data[iy*Nx + ix], data[index] );
        }
    }


    /**
     * checks if the imaginary parts are all 0 for debugging purposes
     **/
    template< class T_COMPLEX >
    void checkIfReal
    (
        const T_COMPLEX * const & rData,
        const unsigned & rnElements
    )
    {
        float avgRe = 0;
        float avgIm = 0;

        for ( unsigned i = 0; i < rnElements; ++i )
        {
            avgRe += fabs( rData[i][0] );
            avgIm += fabs( rData[i][1] );
        }

        avgRe /= (float) rnElements;
        avgIm /= (float) rnElements;

        std::cout << std::scientific
                  << "Avg. Re = " << avgRe << "\n"
                  << "Avg. Im = " << avgIm << "\n";
        assert( avgIm <  1e-5 );
    }


    template< class T_PREC >
    float compareCpuWithGpuArray
    (
        const T_PREC * const & rpData,
        const T_PREC * const & rdpData,
        const unsigned & rnElements
    )
    {
        /* copy data from GPU in order to compare it */
        const unsigned nBytes = rnElements * sizeof(T_PREC);
        const T_PREC * const vec1 = rpData;
        T_PREC * const vec2 = (T_PREC*) malloc( nBytes );
        CUDA_ERROR( cudaMemcpy( (void*) vec2, (void*) rdpData, nBytes, cudaMemcpyDeviceToHost ) );

        float relErr = 0;

        //#pragma omp parallel for reduction( + : relErr )
        for ( unsigned i = 0; i < rnElements; ++i )
        {
            float max = fmax( fabs(vec1[i]), fabs(vec2[i]) );
            /* ignore 0/0 if both are equal and 0 */
            if ( max == 0 )
                max = 1;
            relErr += fabs( vec1[i] - vec2[i] ); // / max;
            //if ( i < 10 )
            //    std::cout << "    " << vec1[i] << " <-> " << vec2[i] << "\n";
        }

        free( vec2 );
        return relErr / rnElements;
    }


    /**
     *
     * In contrast to the normal hybrid input output this function takes
     * pointers to memory buffers instead of allocating them itself.
     * Furthermore it doesn't touch rIntensity and it returns F instead of f
     * in curData.
     * It also doesn't bother to calculate the error at each step.
     *
     * @param[in] rIntensity real measured intensity without phase
     * @param[in] rIntensityFirstGuess first guess for the phase of the
     *            intensity, e.g. a random phase
     * @param[in] gPrevious this isn't actually a guess for the object f, but
     *            an intermediary result for the HIO algorithm. For the first
     *            call it should be equal to g' = IFT[G == rIntensityFirstGuess]
     **/
    int cudaShrinkWrap
    (
        float * const & rIntensity,
        const std::vector<unsigned> & rSize,
        unsigned rnCycles,
        float rTargetError,
        float rHioBeta,
        float rIntensityCutOffAutoCorel,
        float rIntensityCutOff,
        float rSigma0,
        float rSigmaChange,
        unsigned rnHioCycles,
        unsigned rnCores
    )
    {
        if ( rSize.size() != 2 ) return 1;
        const unsigned & Ny = rSize[1];
        const unsigned & Nx = rSize[0];

        /* load libraries and functions which we need */
        using namespace imresh::algorithms;

        /* Evaluate input parameters and fill with default values if necessary */
        if ( rIntensity == NULL ) return 1;
        if ( rTargetError              <= 0 ) rTargetError              = 1e-5;
        if ( rnHioCycles               == 0 ) rnHioCycles               = 20;
        if ( rHioBeta                  <= 0 ) rHioBeta                  = 0.9;
        if ( rIntensityCutOffAutoCorel <= 0 ) rIntensityCutOffAutoCorel = 0.04;
        if ( rIntensityCutOff          <= 0 ) rIntensityCutOff          = 0.2;
        if ( rSigma0                   <= 0 ) rSigma0                   = 3.0;
        if ( rSigmaChange              <= 0 ) rSigmaChange              = 0.01;

        float sigma = rSigma0;

        /* calculate this (length of array) often needed value */
        unsigned nElements = 1;
        for ( unsigned i = 0; i < rSize.size(); ++i )
        {
            assert( rSize[i] > 0 );
            nElements *= rSize[i];
        }

        /* allocate needed memory so that HIO doesn't need to allocate and
         * deallocate on each call */
        cufftComplex * dpCurData;
        cufftComplex * dpgPrevious;
        float * dpIntensity;
        float * dpIsMasked;
        CUDA_ERROR( cudaMalloc( (void**)&dpCurData  , sizeof(dpCurData  [0])*nElements ) );
        CUDA_ERROR( cudaMalloc( (void**)&dpgPrevious, sizeof(dpgPrevious[0])*nElements ) );
        CUDA_ERROR( cudaMalloc( (void**)&dpIntensity, sizeof(dpIntensity[0])*nElements ) );
        CUDA_ERROR( cudaMalloc( (void**)&dpIsMasked , sizeof(dpIsMasked [0])*nElements ) );
        CUDA_ERROR( cudaMemcpy( dpIntensity, rIntensity, sizeof(dpIntensity[0])*nElements, cudaMemcpyHostToDevice ) );

        /* create fft plans G' to g' and g to G */
        cufftHandle ftPlan;
        cufftPlan2d( &ftPlan, Nx, Ny, CUFFT_C2C );

        /* create first guess for mask from autocorrelation (fourier transform
         * of the intensity @see
         * https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem */
        const unsigned nThreads = 512;
        const unsigned nBlocks  = ceil( (float) nElements / nThreads );
        cudaKernelCopyToRealPart<<< nBlocks, nThreads >>>( dpCurData, dpIntensity, nElements );

        cufftExecC2C( ftPlan, dpCurData, dpCurData, CUFFT_INVERSE );
        cudaKernelComplexNormElementwise<<< nBlocks, nThreads >>>( dpIsMasked, dpCurData, nElements );
        //cudaFftShift( dpIsMasked, Nx,Ny );
        cudaGaussianBlur( dpIsMasked, Nx, Ny, sigma );

        /* apply threshold to make binary mask */
        const float maskedAbsMax = cudaReduce( dpIsMasked, nElements, maxFunctor, 0.0f );
        const float maskedThreshold = rIntensityCutOffAutoCorel * maskedAbsMax;
        cudaKernelCutOff<<<nBlocks,nThreads>>>( dpIsMasked, nElements, maskedThreshold, 1.0f, 0.0f );

        /* copy original image into fftw_complex array @todo: add random phase */
        cudaKernelCopyToRealPart<<< nBlocks, nThreads >>>( dpCurData, dpIntensity, nElements );

        /* in the first step the last value for g is to be approximated
         * by g'. The last value for g, called g_k is needed, because
         * g_{k+1} = g_k - hioBeta * g' ! This is inside the loop
         * because the fft is needed */
        cudaMemcpy( dpgPrevious, dpCurData, sizeof(dpCurData[0]) * nElements,
                    cudaMemcpyDeviceToDevice );

        /* repeatedly call HIO algorithm and change mask */
        for ( unsigned iCycleShrinkWrap = 0; iCycleShrinkWrap < rnCycles; ++iCycleShrinkWrap )
        {
            /************************** Update Mask ***************************/
            std::cout << "Update Mask with sigma=" << sigma << "\n";

            /* blur |g'| (normally g' should be real!, so |.| not necessary) */
            cudaKernelComplexNormElementwise<<<nBlocks,nThreads>>>( dpIsMasked, dpCurData, nElements );
            cudaGaussianBlur( dpIsMasked, Nx, Ny, sigma );

            /* apply threshold to make binary mask */
            const float absMax = cudaReduce( dpIsMasked, nElements, maxFunctor, 0.0f );
            const float threshold = rIntensityCutOff * absMax;
            cudaKernelCutOff<<<nBlocks,nThreads>>>( dpIsMasked, nElements, threshold, 1.0f, 0.0f );

            /* update the blurring sigma */
            sigma = fmax( 1.5f, ( 1.0f - rSigmaChange ) * sigma );

            for ( unsigned iHioCycle = 0; iHioCycle < rnHioCycles; ++iHioCycle )
            {
                /* apply domain constraints to g' to get g */
                cudaKernelApplyHioDomainConstraints<<<nBlocks,nThreads>>>
                    ( dpgPrevious, dpCurData, dpIsMasked, nElements, rHioBeta );

                /* Transform new guess g for f back into frequency space G' */
                cufftExecC2C( ftPlan, dpgPrevious, dpCurData, CUFFT_FORWARD );

                /* Replace absolute of G' with measured absolute |F| */
                cudaKernelApplyComplexModulus<<<nBlocks,nThreads>>>
                    ( dpCurData, dpCurData, dpIntensity, nElements );

                cufftExecC2C( ftPlan, dpCurData, dpCurData, CUFFT_INVERSE );
            } // HIO loop

            /* check if we are done */
            const float currentError = calculateHioError( dpCurData /*g'*/, dpIsMasked, nElements );
            std::cout << "[Error " << currentError << "/" << rTargetError << "] "
                      << "[Cycle " << iCycleShrinkWrap << "/" << rnCycles-1 << "]"
                      << "\n";
            if ( rTargetError > 0 && currentError < rTargetError )
                break;
            if ( iCycleShrinkWrap >= rnCycles )
                break;
        } // shrink wrap loop
        cudaKernelCopyFromRealPart<<< nBlocks, nThreads >>>( dpIntensity, dpCurData, nElements );
        CUDA_ERROR( cudaMemcpy( rIntensity, dpIntensity, sizeof(rIntensity[0])*nElements, cudaMemcpyDeviceToHost ) );

        /* free buffers and plans */
        cufftDestroy( ftPlan );
        CUDA_ERROR( cudaFree( dpCurData   ) );
        CUDA_ERROR( cudaFree( dpgPrevious ) );
        CUDA_ERROR( cudaFree( dpIntensity ) );
        CUDA_ERROR( cudaFree( dpIsMasked  ) );

        return 0;
    }

    /**
     * Same as cudaShrinkWrap but with support for async calls.
     */
    int shrinkWrap
    (
        float* const& rIntensity,
        const std::pair<unsigned,unsigned>& rSize,
        cudaStream_t strm,
        unsigned rnCycles,
        float rTargetError,
        float rHioBeta,
        float rIntensityCutOffAutoCorel,
        float rIntensityCutOff,
        float rSigma0,
        float rSigmaChange,
        unsigned rnHioCycles,
        unsigned rnCores
    )
    {
        const unsigned& Ny = rSize.second;
        const unsigned& Nx = rSize.first;

        /* load libraries and functions which we need */
        using namespace imresh::algorithms;

        /* Evaluate input parameters and fill with default values if necessary */
        if ( rIntensity == NULL ) return 1;
        if ( rTargetError              <= 0 ) rTargetError              = 1e-5;
        if ( rnHioCycles               == 0 ) rnHioCycles               = 20;
        if ( rHioBeta                  <= 0 ) rHioBeta                  = 0.9;
        if ( rIntensityCutOffAutoCorel <= 0 ) rIntensityCutOffAutoCorel = 0.04;
        if ( rIntensityCutOff          <= 0 ) rIntensityCutOff          = 0.2;
        if ( rSigma0                   <= 0 ) rSigma0                   = 3.0;
        if ( rSigmaChange              <= 0 ) rSigmaChange              = 0.01;

        float sigma = rSigma0;

        /* calculate this (length of array) often needed value */
        assert( rSize.first > 0 && rSize.second > 0 );
        unsigned int nElements = 1 * rSize.first * rSize.second;

        /* allocate needed memory so that HIO doesn't need to allocate and
         * deallocate on each call */
        cufftComplex * dpCurData;
        cufftComplex * dpgPrevious;
        float * dpIntensity;
        float * dpIsMasked;
        CUDA_ERROR( cudaMalloc( (void**)&dpCurData  , sizeof(dpCurData  [0])*nElements ) );
        CUDA_ERROR( cudaMalloc( (void**)&dpgPrevious, sizeof(dpgPrevious[0])*nElements ) );
        CUDA_ERROR( cudaMalloc( (void**)&dpIntensity, sizeof(dpIntensity[0])*nElements ) );
        CUDA_ERROR( cudaMalloc( (void**)&dpIsMasked , sizeof(dpIsMasked [0])*nElements ) );
        CUDA_ERROR( cudaMemcpyAsync( dpIntensity, rIntensity, sizeof(dpIntensity[0])*nElements, cudaMemcpyHostToDevice, strm ) );

        /* create fft plans G' to g' and g to G */
        cufftHandle ftPlan;
        cufftPlan2d( &ftPlan, Nx, Ny, CUFFT_C2C );

        /* create first guess for mask from autocorrelation (fourier transform
         * of the intensity @see
         * https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem */
        const unsigned nThreads = 512;
        const unsigned nBlocks  = ceil( (float) nElements / nThreads );
        cudaKernelCopyToRealPart<<<nBlocks,nThreads,0,strm>>>( dpCurData, dpIntensity, nElements );

        cufftExecC2C( ftPlan, dpCurData, dpCurData, CUFFT_INVERSE );
        cudaKernelComplexNormElementwise<<<nBlocks,nThreads,0,strm>>>( dpIsMasked, dpCurData, nElements );
        //cudaFftShift( dpIsMasked, Nx,Ny );
        cudaGaussianBlur( dpIsMasked, Nx, Ny, sigma );

        /* apply threshold to make binary mask */
        const float maskedAbsMax = cudaReduce( dpIsMasked, nElements, maxFunctor, 0.0f );
        const float maskedThreshold = rIntensityCutOffAutoCorel * maskedAbsMax;
        cudaKernelCutOff<<<nBlocks,nThreads,0,strm>>>( dpIsMasked, nElements, maskedThreshold, 1.0f, 0.0f );

        /* copy original image into fftw_complex array @todo: add random phase */
        cudaKernelCopyToRealPart<<<nBlocks,nThreads,0,strm>>>( dpCurData, dpIntensity, nElements );

        /* in the first step the last value for g is to be approximated
         * by g'. The last value for g, called g_k is needed, because
         * g_{k+1} = g_k - hioBeta * g' ! This is inside the loop
         * because the fft is needed */
        cudaMemcpyAsync( dpgPrevious, dpCurData, sizeof(dpCurData[0]) * nElements,
                    cudaMemcpyDeviceToDevice, strm );

        /* repeatedly call HIO algorithm and change mask */
        for ( unsigned iCycleShrinkWrap = 0; iCycleShrinkWrap < rnCycles; ++iCycleShrinkWrap )
        {
            /************************** Update Mask ***************************/
#           ifdef IMRESH_DEBUG
                std::cout << "imresh::algorithms::cuda::shrinkWrap(): Update Mask with sigma=" << sigma << std::endl;
#           endif

            /* blur |g'| (normally g' should be real!, so |.| not necessary) */
            cudaKernelComplexNormElementwise<<<nBlocks,nThreads,0,strm>>>( dpIsMasked, dpCurData, nElements );
            cudaGaussianBlur( dpIsMasked, Nx, Ny, sigma );

            /* apply threshold to make binary mask */
            const float absMax = cudaReduce( dpIsMasked, nElements, maxFunctor, 0.0f );
            const float threshold = rIntensityCutOff * absMax;
            cudaKernelCutOff<<<nBlocks,nThreads,0,strm>>>( dpIsMasked, nElements, threshold, 1.0f, 0.0f );

            /* update the blurring sigma */
            sigma = fmax( 1.5f, ( 1.0f - rSigmaChange ) * sigma );

            for ( unsigned iHioCycle = 0; iHioCycle < rnHioCycles; ++iHioCycle )
            {
                /* apply domain constraints to g' to get g */
                cudaKernelApplyHioDomainConstraints<<<nBlocks,nThreads,0,strm>>>
                    ( dpgPrevious, dpCurData, dpIsMasked, nElements, rHioBeta );

                /* Transform new guess g for f back into frequency space G' */
                cufftExecC2C( ftPlan, dpgPrevious, dpCurData, CUFFT_FORWARD );

                /* Replace absolute of G' with measured absolute |F| */
                cudaKernelApplyComplexModulus<<<nBlocks,nThreads,0,strm>>>
                    ( dpCurData, dpCurData, dpIntensity, nElements );

                cufftExecC2C( ftPlan, dpCurData, dpCurData, CUFFT_INVERSE );
            } // HIO loop

            /* check if we are done */
            const float currentError = calculateHioError( dpCurData /*g'*/, dpIsMasked, nElements );
#           ifdef IMRESH_DEBUG
                std::cout << "imresh::algorithms::cuda::shrinkWrap(): [Error " <<
                    currentError << "/" << rTargetError << "] "
                    << "[Cycle " << iCycleShrinkWrap << "/" << rnCycles-1 << "]"
                    << std::endl;
#           endif
            if ( rTargetError > 0 && currentError < rTargetError )
                break;
            if ( iCycleShrinkWrap >= rnCycles )
                break;
        } // shrink wrap loop
        cudaKernelCopyFromRealPart<<<nBlocks,nThreads,0,strm>>>( dpIntensity, dpCurData, nElements );
        CUDA_ERROR( cudaMemcpyAsync( rIntensity, dpIntensity, sizeof(rIntensity[0])*nElements, cudaMemcpyDeviceToHost, strm ) );

        /* free buffers and plans */
        cufftDestroy( ftPlan );
        CUDA_ERROR( cudaFree( dpCurData   ) );
        CUDA_ERROR( cudaFree( dpgPrevious ) );
        CUDA_ERROR( cudaFree( dpIntensity ) );
        CUDA_ERROR( cudaFree( dpIsMasked  ) );

        return 0;
    }
} // namespace cuda
} // namespace algorithms
} // namespace imresh
