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

#ifndef NDEBUG // change the following, if you want to turn on debugging
#   define DEBUG_CUDASHRINKWRAP 0
#   define WRITE_OUT_SHRINKWRAP_DEBUG 0
#else   // leave this as it is
#   define DEBUG_CUDASHRINKWRAP 0
#   define WRITE_OUT_SHRINKWRAP_DEBUG 0
#endif

#include <iostream>
#include <cstddef>      // NULL
#include <cstring>      // memcpy
#include <cassert>
#include <cmath>
#include <vector>
#include <cuda.h>       // atomicCAS
#include <cufft.h>
#include <utility>      // std::pair
#include "algorithms/cuda/cudaGaussian.hpp"
#include "algorithms/cuda/cudaVectorReduce.hpp"
#if DEBUG_CUDASHRINKWRAP == 1
#   include "algorithms/vectorReduce.hpp"
#   include "algorithms/vectorElementwise.hpp"
#endif
#if WRITE_OUT_SHRINKWRAP_DEBUG == 1
#   include <sstream>
#   include "io/writeOutFuncs/writeOutFuncs.hpp"
#endif
#include "libs/cudacommon.hpp"
#include "libs/checkCufftError.hpp"
#include "cudaVectorElementwise.hpp"


namespace imresh
{
namespace algorithms
{
namespace cuda
{


#   ifdef IMRESH_DEBUG
        /**
         * checks if the imaginary parts are all 0. For debugging purposes
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
#   endif


    template< class T_PREC >
    float compareCpuWithGpuArray
    (
        T_PREC const * const __restrict__ rpData,
        T_PREC const * const __restrict__ rdpData,
        unsigned int const rnElements
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


    int cudaShrinkWrap
    (
        float *      const  rIntensity               ,
        unsigned int const  rImageWidth              ,
        unsigned int const  rImageHeight             ,
        cudaStream_t const  rStream                  ,
        unsigned int        nBlocks                  ,
        unsigned int        nThreads                 ,
        unsigned int        rnCycles                 ,
        float               rTargetError             ,
        float               rHioBeta                 ,
        float               rIntensityCutOffAutoCorel,
        float               rIntensityCutOff         ,
        float               rSigma0                  ,
        float               rSigmaChange             ,
        unsigned int        rnHioCycles
    )
    {
        /* load libraries and functions which we need */
        using namespace imresh::algorithms;
        using imresh::libs::mallocCudaArray;

        /* Evaluate input parameters and fill with default values if necessary */
        assert( rImageWidth > 0 );
        assert( rImageHeight > 0 );
        assert( rIntensity != NULL );
        /* this makes it possible to specifiy new values for e.g. rSigma0,
         * while still using the default values for rHioBeta, rTargetError,
         * ... */
        if ( rnCycles                  == 0 ) rnCycles                  = 20  ;
        if ( rTargetError              <= 0 ) rTargetError              = 1e-5;
        if ( rHioBeta                  <= 0 ) rHioBeta                  = 0.9 ;
        if ( rIntensityCutOffAutoCorel <= 0 ) rIntensityCutOffAutoCorel = 0.02; /* better: choose a value such that the mask contains 50% of the image, or some over scheme -> try making a histogram @TODO!!! */
        if ( rIntensityCutOff          <= 0 ) rIntensityCutOff          = 0.2 ;
        if ( rSigma0                   <= 0 ) rSigma0                   = 3.0 ;
        if ( rSigmaChange              <= 0 ) rSigmaChange              = 0.01;
        if ( rnHioCycles               == 0 ) rnHioCycles               = 20  ;

        float sigma = rSigma0;
        auto const nElements = rImageWidth * rImageHeight;

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
        CUDA_ERROR( cudaMemcpyAsync( dpIntensity, rIntensity, sizeof(dpIntensity[0])*nElements, cudaMemcpyHostToDevice, rStream ) );
        #if WRITE_OUT_SHRINKWRAP_DEBUG == 1
            /* allocate 2*nElements to hold if necessary nElements
             * cufftComplex elements */
            float * pHostToWrite = new float[ 2*nElements ];
        #endif

        #define RUN(x) cudaKernel##x<<<nBlocks,nThreads,0,rStream>>>

        /* create fft plans G' to g' and g to G */
        cufftHandle ftPlan;
        CUFFT_ERROR( cufftPlan2d( &ftPlan, rImageHeight /* nRows */, rImageWidth /* nColumns */, CUFFT_C2C ) );
        CUFFT_ERROR( cufftSetStream( ftPlan, rStream ) );

        RUN(CopyToRealPart)( dpCurData, dpIntensity, nElements );
        CUDA_ERROR( cudaPeekAtLastError() );
        /* intensity -> autocorrelation / G (current guess for object) */
        CUFFT_ERROR( cufftExecC2C( ftPlan, dpCurData, dpCurData, CUFFT_INVERSE ) );

        /* repeatedly call HIO algorithm and change mask */
        for ( auto iCycleShrinkWrap = 0u; iCycleShrinkWrap < rnCycles; ++iCycleShrinkWrap )
        {
            /************************** Update Mask ***************************/

            /* blur |g'| (normally g' should be real!, so |.| not necessary, only copy from real part to dpIsMasked) */
            RUN(ComplexNormElementwise)( dpIsMasked, dpCurData, nElements );
            CUDA_ERROR( cudaPeekAtLastError() );
            cudaGaussianBlur( dpIsMasked, rImageWidth, rImageHeight, sigma, rStream, true /* don't call cudaDeviceSynchronize */ );
            #if WRITE_OUT_SHRINKWRAP_DEBUG == 1
            {
                std::stringstream fname;
                fname << "shrinkWrap_a_first-mask-blurred.png";
                CUDA_ERROR( cudaStreamSynchronize( rStream ) );
                CUDA_ERROR( cudaMemcpy( pHostToWrite, dpIsMasked, nElements *
                    sizeof( pHostToWrite[0] ), cudaMemcpyDeviceToHost ) );
                imresh::io::writeOutFuncs::writeOutPNG(
                    pHostToWrite, rImageWidth, rImageHeight, fname.str().c_str()
                );
            }
            {
                std::stringstream fname;
                fname << "shrinkWrap_a_first-mask-blurred-log-scale.png";
                CUDA_ERROR( cudaStreamSynchronize( rStream ) );
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
            /* create first guess for mask from autocorrelation (fourier transform
             * of the intensity @see
             * https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem */
            const float absMax = cudaVectorMax( dpIsMasked, nElements, rStream );
            const float threshold = ( iCycleShrinkWrap == 0 ?
                rIntensityCutOffAutoCorel : rIntensityCutOff ) * absMax;
            RUN(CutOff)( dpIsMasked, nElements, threshold, 1.0f, 0.0f );
            CUDA_ERROR( cudaPeekAtLastError() );
            #if WRITE_OUT_SHRINKWRAP_DEBUG == 1
            {
                std::stringstream fname;
                fname << "shrinkWrap_b_iC-" << iCycleShrinkWrap
                      << "-a_mask.png";
                CUDA_ERROR( cudaStreamSynchronize( rStream ) );
                CUDA_ERROR( cudaMemcpy( pHostToWrite, dpIsMasked, nElements *
                    sizeof( pHostToWrite[0] ), cudaMemcpyDeviceToHost ) );
                imresh::io::writeOutFuncs::writeOutPNG(
                    pHostToWrite, rImageWidth, rImageHeight, fname.str().c_str()
                );
            }
            #endif

            /* update the blurring sigma */
            sigma = fmax( 1.5f, ( 1.0f - rSigmaChange ) * sigma );
            /************************ Update Mask End *************************/

            /* in first step add random phase to intensity by filling
             * unmasked area with ranfom uniform noise with mean value
             * equal to zero frequency / highest value in diffraction pattern */
            /* in the first step the last value for g is to be approximated
             * by g'. The last value for g, called g_k is needed, because
             * g_{k+1} = g_k - hioBeta * g' ! This is inside the loop
             * because the fft is needed -> dpgPrevious */
            if ( iCycleShrinkWrap == 0 )
            {
                //RUN(ApplyComplexModulus)( dpCurData, dpCurData, dpIntensity, nElements );
                cudaMemcpyAsync( dpgPrevious, dpCurData, sizeof(dpCurData[0]) * nElements,
                            cudaMemcpyDeviceToDevice, rStream );
            }

            for ( auto iHioCycle = 0u; iHioCycle < rnHioCycles; ++iHioCycle )
            {
                /* apply domain constraints to g' to get g */
                RUN(ApplyHioDomainConstraints)
                    ( dpgPrevious, dpCurData, dpIsMasked, nElements, rHioBeta );
                CUDA_ERROR( cudaPeekAtLastError() );

                /* Transform new guess g for f back into frequency space G' */
                CUFFT_ERROR( cufftExecC2C( ftPlan, dpgPrevious, dpCurData, CUFFT_FORWARD ) );

                /* Replace absolute of G' with measured absolute |F| */
                RUN(ApplyComplexModulus)( dpCurData, dpCurData, dpIntensity, nElements );
                CUDA_ERROR( cudaPeekAtLastError() );
                #if WRITE_OUT_SHRINKWRAP_DEBUG == 1
                {
                    std::stringstream fname;
                    fname << "shrinkWrap_b_iC-" << iCycleShrinkWrap
                          << "_iHio-" << iHioCycle << "-b_intensity.png";
                    CUDA_ERROR( cudaStreamSynchronize( rStream ) );
                    CUDA_ERROR( cudaMemcpy( pHostToWrite, dpCurData, 2*nElements *
                        sizeof( pHostToWrite[0] ), cudaMemcpyDeviceToHost ) );
                    for ( auto i = 0u; i < nElements; ++i )
                    {
                        pHostToWrite[i] = sqrtf( pHostToWrite[2*i+0]*pHostToWrite[2*i+0] +
                                                 pHostToWrite[2*i+1]*pHostToWrite[2*i+1] );
                    }
                    imresh::io::writeOutFuncs::writeOutPNG(
                        pHostToWrite, rImageWidth, rImageHeight, fname.str().c_str()
                    );
                }
                #endif

                CUFFT_ERROR( cufftExecC2C( ftPlan, dpCurData, dpCurData, CUFFT_INVERSE ) );
                #if WRITE_OUT_SHRINKWRAP_DEBUG == 1
                {
                    std::stringstream fname;
                    fname << "shrinkWrap_b_iC-" << iCycleShrinkWrap
                          << "_iHio-" << iHioCycle << "-c_object.png";
                    CUDA_ERROR( cudaStreamSynchronize( rStream ) );
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
            const float currentError = cudaCalculateHioError( dpCurData /*g'*/, dpIsMasked, nElements, false /* don't invert mask */, rStream );
            #ifdef IMRESH_DEBUG
                std::cout << "[Error " << currentError << "/" << rTargetError << "] "
                          << "[Cycle " << iCycleShrinkWrap << "/" << rnCycles-1 << "]"
                          << std::endl;
            #endif
            if ( rTargetError > 0 && currentError < rTargetError )
                break;
            if ( iCycleShrinkWrap >= rnCycles )
                break;

            #if WRITE_OUT_SHRINKWRAP_DEBUG == 1
            {
                std::stringstream fname;
                fname << "shrinkWrap_b_iC-" << iCycleShrinkWrap
                      << "-d_object.png";
                CUDA_ERROR( cudaStreamSynchronize( rStream ) );
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
        RUN(CopyFromRealPart)( dpIntensity, dpCurData, nElements );
        CUDA_ERROR( cudaPeekAtLastError() );
        CUDA_ERROR( cudaMemcpyAsync( rIntensity, dpIntensity, sizeof(rIntensity[0])*nElements, cudaMemcpyDeviceToHost, rStream ) );
        #undef RUN

        /* wait for everything to finish */
        CUDA_ERROR( cudaStreamSynchronize( rStream ) );

        /* free buffers and plans */
        CUFFT_ERROR( cufftDestroy( ftPlan ) );
        CUDA_ERROR( cudaFree( dpCurData   ) );
        CUDA_ERROR( cudaFree( dpgPrevious ) );
        CUDA_ERROR( cudaFree( dpIntensity ) );
        CUDA_ERROR( cudaFree( dpIsMasked  ) );
        #if WRITE_OUT_SHRINKWRAP_DEBUG == 1
            delete[] pHostToWrite;
        #endif

        return 0;
    }


} // namespace cuda
} // namespace algorithms
} // namespace imresh
