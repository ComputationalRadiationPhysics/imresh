/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2016 Maximilian Knespel, Phillip Trommler
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


#include "cudaShrinkWrap.hpp"


#include <iostream>
#include <cstddef>              // NULL
#include <cassert>
#include <cuda_to_cupla.hpp>    // atomicCAS, cudaMemcpy
#include "libs/cufft_to_cupla.hpp"   // cufftComplex
#include "algorithms/cuda/cudaGaussian.hpp"
#include "algorithms/cuda/cudaVectorReduce.hpp"
#if DEBUG_CUDASHRINKWRAP == 1
#   include "io/writeOutFuncs/writeOutFuncs.hpp"
#   define WRITE_OUT_SHRINKWRAP_DEBUG 1
#endif
#include "libs/cudacommon.hpp"  // CUDA_ERROR
#include "libs/CudaKernelConfig.hpp"
#include "cudaVectorElementwise.hpp"


namespace imresh
{
namespace algorithms
{
namespace cuda
{


    int cudaShrinkWrap
    (
        libs::CudaKernelConfig rKernelConfig            ,
        float *      const     rIoData                  ,
        unsigned int const     rImageWidth              ,
        unsigned int const     rImageHeight             ,
        unsigned int           rnCycles                 ,
        float                  rTargetError             ,
        float                  rHioBeta                 ,
        float                  rIntensityCutOffAutoCorel,
        float                  rIntensityCutOff         ,
        float                  rSigma0                  ,
        float                  rSigmaChange             ,
        unsigned int           rnHioCycles
    )
    {
        /* load libraries and functions which we need */
        using namespace imresh::algorithms;
        using imresh::libs::mallocCudaArray;
        using namespace haLT; // types::Vec2, types::Complex, FFT_Definition, ...

        rKernelConfig.check();
        auto const & rStream  = rKernelConfig.iStream;

        /* Evaluate input parameters and fill with default values if necessary */
        assert( rImageWidth  > 0 );
        assert( rImageHeight > 0 );
        assert( rIoData != NULL );
        /* this makes it possible to specifiy new values for e.g. rSigma0,
         * while still using the default values for rHioBeta, rTargetError,
         * ... */
        if ( rTargetError              <= 0 ) rTargetError              = 1e-5 ;
        if ( rnHioCycles               == 0 ) rnHioCycles               = 20   ;
        if ( rHioBeta                  <= 0 ) rHioBeta                  = 0.9  ;
        if ( rIntensityCutOffAutoCorel <= 0 ) rIntensityCutOffAutoCorel = 0.04 ;
        if ( rIntensityCutOff          <= 0 ) rIntensityCutOff          = 0.2  ;
        if ( rSigma0                   <= 0 ) rSigma0                   = 3.0  ;
        if ( rSigmaChange              <= 0 ) rSigmaChange              = 0.01 ;

        float sigma = rSigma0;
        unsigned const nElements = rImageWidth * rImageHeight;

        /* allocate needed memory so that HIO doesn't need to allocate and
         * deallocate on each call */
        cufftComplex * dpCurData;
        cufftComplex * dpgPrevious;
        float * dpIntensity;
        float * dpIsMasked;
        mallocCudaArray( &dpCurData  , nElements );
        mallocCudaArray( &dpgPrevious, nElements );
        mallocCudaArray( &dpIntensity, nElements );
        mallocCudaArray( &dpIsMasked , nElements );
        CUDA_ERROR( cudaMemcpyAsync( dpIntensity, rIoData,
            sizeof(dpIntensity[0])*nElements, cudaMemcpyHostToDevice, rStream ) );

        /* shorthand for HaLT wrapper */
        auto wcdp /* wrapComplexDevicePointer */ =
            [ &rImageHeight,  &rImageWidth ]( cufftComplex * const & rdp )
            {
                auto arraySize = types::Vec2
                                 {
                                    rImageHeight /* Ny, nRows */,
                                    rImageWidth  /* Nx, nCols */
                                 };
                return mem::wrapPtr
                       <
                           true /* is complex */,
                           true /* is device pointer */
                       >( (types::Complex<float> *) rdp, arraySize );
            };

        #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
            /* allocate 2*nElements to hold if necessary nElements
             * cufftComplex elements */
            float * pHostToWrite = new float[2*nElements];
        #endif

        /* create plan and wrap data g->G (dpgPrevious->dpCurData) */
        using Plan_gToG = FFT_Definition<
            FFT_Kind::Complex2Complex,
            2             ,  /* dims         */
            float         ,  /* precision    */
            std::true_type,  /* forward      */
            false            /* not in-place */
        >;
        auto dpgPreviousIn = Plan_gToG::wrapInput ( wcdp( dpgPrevious ) );
        auto dpCurDataOut  = Plan_gToG::wrapOutput( wcdp( dpCurData   ) );
        auto fft_gToG = makeFftPlan( dpgPreviousIn, dpCurDataOut );

        /* create plan and wrap data G'->g' (dpCurData->dpCurData) */
        using Plan_GPrimeTogPrime = FFT_Definition<
            FFT_Kind::Complex2Complex,
            2              , /* dims         */
            float          , /* precision    */
            std::false_type, /* inverse      */
            true             /* in-place     */
        >;
        auto dpCurDataIn = Plan_GPrimeTogPrime::wrapInput ( wcdp( dpCurData ) );
        auto fft_GPrimeTogPrime = makeFftPlan( dpCurDataIn );

        //#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        //    cufftSetStream( ftPlan, rStream );
        //#endif

        /* create first guess for mask from autocorrelation (fourier transform
         * of the intensity @see
         * https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem */
        cudaCopyToRealPart( rKernelConfig, dpCurData, dpIntensity, nElements );
            #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
            {
                std::stringstream fname;
                fname << "shrinkWrap_original-object.png";
                CUDA_ERROR( cudaMemcpy2D(
                    pHostToWrite,              /* dest addr */
                    sizeof( pHostToWrite[0] ), /* dest pitch */
                    dpCurData,                 /* src addr */
                    sizeof( dpCurData[0] ),    /* src pitch */
                    sizeof( pHostToWrite[0] ), /* col width (bytes) */
                    nElements,                 /* number of columns */
                    cudaMemcpyDeviceToHost
                ) );
                imresh::io::writeOutFuncs::writeOutPNG(
                    pHostToWrite, rImageWidth, rImageHeight, fname.str().c_str()
                );
            }
            #endif
        fft_GPrimeTogPrime( dpCurDataIn );

            #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
            {
                std::stringstream fname;
                fname << "shrinkWrap_original-object-fft.png";
                CUDA_ERROR( cudaMemcpy2D(
                    pHostToWrite,              /* dest addr */
                    sizeof( pHostToWrite[0] ), /* dest pitch */
                    dpCurData,                 /* src addr */
                    sizeof( dpCurData[0] ),    /* src pitch */
                    sizeof( pHostToWrite[0] ), /* col width (bytes) */
                    nElements,                 /* number of columns */
                    cudaMemcpyDeviceToHost
                ) );
                imresh::io::writeOutFuncs::writeOutPNG(
                    pHostToWrite, rImageWidth, rImageHeight, fname.str().c_str()
                );
            }
            #endif

        cudaComplexNormElementwise( rKernelConfig, dpIsMasked, dpCurData, nElements );

            #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
            {
                std::stringstream fname;
                fname << "shrinkWrap_original-object-fft-norm.png";
                CUDA_ERROR( cudaMemcpy( pHostToWrite, dpIsMasked, nElements *
                    sizeof( pHostToWrite[0] ), cudaMemcpyDeviceToHost ) );
                imresh::io::writeOutFuncs::writeOutPNG(
                    pHostToWrite, rImageWidth, rImageHeight, fname.str().c_str()
                );
            }
            #endif

        cudaGaussianBlur( dpIsMasked, rImageWidth, rImageHeight, sigma, rStream,
                          true /* don't call cudaDeviceSynchronize */ );

            #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
            {
                std::stringstream fname;
                fname << "shrinkWrap_first-mask-blurred.png";
                CUDA_ERROR( cudaMemcpy( pHostToWrite, dpIsMasked, nElements *
                    sizeof( pHostToWrite[0] ), cudaMemcpyDeviceToHost ) );
                imresh::io::writeOutFuncs::writeOutPNG(
                    pHostToWrite, rImageWidth, rImageHeight, fname.str().c_str()
                );
            }
            {
                std::stringstream fname;
                fname << "shrinkWrap_first-mask-blurred-log-scale.png";

                CUDA_ERROR( cudaMemcpy( pHostToWrite, dpIsMasked, nElements *
                    sizeof( pHostToWrite[0] ), cudaMemcpyDeviceToHost ) );
                #pragma omp parallel for
                for ( auto i = 0u; i < nElements; ++i )
                    pHostToWrite[i] = logf( pHostToWrite[i] );

                imresh::io::writeOutFuncs::writeOutPNG(
                    pHostToWrite, rImageWidth, rImageHeight, fname.str().c_str()
                );
            }
            #endif
        /* apply threshold to make binary mask */
        float const maskedAbsMax = cudaVectorMax( rKernelConfig, dpIsMasked, nElements );
        float const maskedThreshold = rIntensityCutOffAutoCorel * maskedAbsMax;
        cudaCutOff( rKernelConfig, dpIsMasked, nElements, maskedThreshold, 1.0f, 0.0f );

        /* copy original image into fftw_complex array @todo: add random phase */
        cudaCopyToRealPart( rKernelConfig, dpCurData, dpIntensity, nElements );

        /* in the first step the last value for g is to be approximated
         * by g'. The last value for g, called g_k is needed, because
         * g_{k+1} = g_k - hioBeta * g' ! This is inside the loop
         * because the fft is needed */
        cudaMemcpyAsync( dpgPrevious, dpCurData, sizeof(dpCurData[0]) * nElements,
                    cudaMemcpyDeviceToDevice, rStream );

        /* repeatedly call HIO algorithm and change mask */
        for ( auto iCycleShrinkWrap = 0u; iCycleShrinkWrap < rnCycles; ++iCycleShrinkWrap )
        {
            /************************** Update Mask ***************************/

            /* blur |g'| (normally g' should be real!, so |.| not necessary) */
            cudaComplexNormElementwise( rKernelConfig, dpIsMasked, dpCurData, nElements );
            cudaGaussianBlur( dpIsMasked, rImageWidth, rImageHeight, sigma, rStream, true /* don't call cudaDeviceSynchronize */ );

            #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
            {
                std::stringstream fname;
                fname << "shrinkWrap_iC-" << iCycleShrinkWrap
                      << "-mask-blurred.png";
                CUDA_ERROR( cudaMemcpy( pHostToWrite, dpIsMasked, nElements *
                    sizeof( pHostToWrite[0] ), cudaMemcpyDeviceToHost ) );
                imresh::io::writeOutFuncs::writeOutPNG(
                    pHostToWrite, rImageWidth, rImageHeight, fname.str().c_str()
                );
            }
            {
                std::stringstream fname;
                fname << "shrinkWrap_iC-" << iCycleShrinkWrap
                      << "-mask-blurred-log-scale.png";

                CUDA_ERROR( cudaMemcpy( pHostToWrite, dpIsMasked, nElements *
                    sizeof( pHostToWrite[0] ), cudaMemcpyDeviceToHost ) );
                #pragma omp parallel for
                for ( auto i = 0u; i < nElements; ++i )
                    pHostToWrite[i] = logf( pHostToWrite[i] );

                imresh::io::writeOutFuncs::writeOutPNG(
                    pHostToWrite, rImageWidth, rImageHeight, fname.str().c_str()
                );
            }
            #endif

            /* apply threshold to make binary mask */
            float const absMax = cudaVectorMax( rKernelConfig, dpIsMasked, nElements );
            float const threshold = rIntensityCutOff * absMax;
            cudaCutOff( rKernelConfig, dpIsMasked, nElements, threshold, 1.0f, 0.0f );

            /* update the blurring sigma */
            sigma = fmax( 1.5f, ( 1.0f - rSigmaChange ) * sigma );

            #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
            {
                std::stringstream fname;
                fname << "shrinkWrap_iC-" << iCycleShrinkWrap
                      << "-mask.png";
                CUDA_ERROR( cudaMemcpy( pHostToWrite, dpIsMasked, nElements *
                    sizeof( pHostToWrite[0] ), cudaMemcpyDeviceToHost ) );
                imresh::io::writeOutFuncs::writeOutPNG(
                    pHostToWrite, rImageWidth, rImageHeight, fname.str().c_str()
                );
            }
            #endif

            for ( unsigned iHioCycle = 0; iHioCycle < rnHioCycles; ++iHioCycle )
            {
                /* apply domain constraints to g' to get g */
                cudaApplyHioDomainConstraints( rKernelConfig, dpgPrevious, dpCurData, dpIsMasked, nElements, rHioBeta );

                /* Transform new guess g for f back into frequency space G */
                fft_gToG( dpgPreviousIn, dpCurDataOut );

                /* Replace absolute of G with measured absolute |F| to get G' */
                cudaApplyComplexModulus( rKernelConfig, dpCurData, dpCurData, dpIntensity, nElements );

                #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
                {
                    std::stringstream fname;
                    fname << "shrinkWrap_iC-" << iCycleShrinkWrap
                          << "_iHio-" << iHioCycle << "-intensity.png";

                    CUDA_ERROR( cudaMemcpy( pHostToWrite, dpIsMasked, 2*nElements *
                        sizeof( pHostToWrite[0] ), cudaMemcpyDeviceToHost ) );
                    #pragma omp parallel for
                    for ( auto i = 0u; i < nElements; ++i )
                        pHostToWrite[i] = sqrtf( pHostToWrite[2*i+0]*pHostToWrite[2*i+0] + pHostToWrite[2*i+1]*pHostToWrite[2*i+1] );

                    imresh::io::writeOutFuncs::writeOutPNG(
                        pHostToWrite, rImageWidth, rImageHeight, fname.str().c_str()
                    );
                }
                #endif

                fft_GPrimeTogPrime( dpCurDataIn );

                #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
                {
                    std::stringstream fname;
                    fname << "shrinkWrap_iC-" << iCycleShrinkWrap
                          << "_iHio-" << iHioCycle << "-object.png";

                    CUDA_ERROR( cudaMemcpy2D(
                        pHostToWrite,              /* dest addr */
                        sizeof( pHostToWrite[0] ), /* dest pitch */
                        dpCurData,                 /* src addr */
                        sizeof( dpCurData[0] ),    /* src pitch */
                        sizeof( pHostToWrite[0] ), /* col width (bytes) */
                        nElements,                 /* number of columns */
                        cudaMemcpyDeviceToHost
                    ) );

                    imresh::io::writeOutFuncs::writeOutPNG(
                        pHostToWrite, rImageWidth, rImageHeight, fname.str().c_str()
                    );
                }
                #endif
            } // HIO loop

            /* check if we are done */
            auto const currentError = cudaCalculateHioError( rKernelConfig,
                dpCurData /*g'*/, dpIsMasked, nElements, false /* don't invert mask */ );
#           ifdef IMRESH_DEBUG
                std::cout << "[Error " << currentError << "/" << rTargetError << "] "
                          << "[Cycle " << iCycleShrinkWrap << "/" << rnCycles-1 << "]"
                          << std::endl;
#           endif
            if ( rTargetError > 0 && currentError < rTargetError )
                break;
            if ( iCycleShrinkWrap >= rnCycles )
                break;

            #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
            {
                std::stringstream fname;
                fname << "shrinkWrap_iC-" << iCycleShrinkWrap
                      << "-object.png";

                CUDA_ERROR( cudaMemcpy2D(
                    pHostToWrite,              /* dest addr */
                    sizeof( pHostToWrite[0] ), /* dest pitch */
                    dpCurData,                 /* src addr */
                    sizeof( dpCurData[0] ),    /* src pitch */
                    sizeof( pHostToWrite[0] ), /* col width (bytes) */
                    nElements,                 /* number of columns */
                    cudaMemcpyDeviceToHost
                ) );

                imresh::io::writeOutFuncs::writeOutPNG(
                    pHostToWrite, rImageWidth, rImageHeight, fname.str().c_str()
                );
            }
            #endif
        } // shrink wrap loop
        cudaCopyFromRealPart( rKernelConfig, dpIntensity, dpCurData, nElements );

        CUDA_ERROR( cudaMemcpyAsync( rIoData, dpIntensity, sizeof(rIoData[0])*nElements, cudaMemcpyDeviceToHost, rStream ) );

        /* wait for everything to finish */
        CUDA_ERROR( cudaStreamSynchronize( rStream ) );

        CUDA_ERROR( cudaFree( dpCurData   ) );
        CUDA_ERROR( cudaFree( dpgPrevious ) );
        CUDA_ERROR( cudaFree( dpIntensity ) );
        CUDA_ERROR( cudaFree( dpIsMasked  ) );

        #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
            delete[] pHostToWrite;
        #endif

        return 0;
    }


} // namespace cuda
} // namespace algorithms
} // namespace imresh
