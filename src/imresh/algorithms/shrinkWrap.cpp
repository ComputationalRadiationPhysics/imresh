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


#define DEBUG_SHRINKWRAPP_CPP 0

#include "shrinkWrap.hpp"

#include <cstddef>    // NULL
#include <cstring>    // memcpy
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>    // setw
#include <string>
#include <fstream>
#include <vector>
#ifdef USE_FFTW
#   include <fftw3.h>
#endif
#include "libs/gaussian.hpp"
#include "libs/hybridInputOutput.hpp" // calculateHioError
#include "algorithms/vectorReduce.hpp"
#include "algorithms/vectorElementwise.hpp"
#if defined( IMRESH_DEBUG ) or ( DEBUG_SHRINKWRAPP_CPP == 1 )
#   ifdef USE_PNG
#       include <sstream>
#       include "io/writeOutFuncs/writeOutFuncs.hpp"
#       define WRITE_OUT_SHRINKWRAP_DEBUG 1
#   endif
#endif


namespace imresh
{
namespace algorithms
{


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

#ifdef USE_FFTW
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


    int shrinkWrap
    (
        float * const rIoData,
        unsigned int const rImageWidth,
        unsigned int const rImageHeight,
        unsigned int rnCycles,
        float rTargetError,
        float rHioBeta,
        float rIntensityCutOffAutoCorel,
        float rIntensityCutOff,
        float rSigma0,
        float rSigmaChange,
        unsigned int rnHioCycles
    )
    {
        auto const rIntensity = rIoData;

        unsigned int const Nx = rImageWidth;
        unsigned int const Ny = rImageHeight;
        unsigned int const nElements = Nx*Ny;   // very often needed

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

        /* allocate needed memory so that HIO doesn't need to allocate and
         * deallocate on each call */
        fftwf_complex * const curData   = fftwf_alloc_complex( nElements );
        fftwf_complex * const gPrevious = fftwf_alloc_complex( nElements );
        auto const isMasked = new float[nElements];

        /* create fft plans G' to g' and g to G */
        std::vector<unsigned int> rSize{ rImageHeight, rImageWidth };
        auto toRealSpace = fftwf_plan_dft( rSize.size(),
            (int*) &rSize[0], curData, curData, FFTW_BACKWARD, FFTW_ESTIMATE );
        auto toFreqSpace = fftwf_plan_dft( rSize.size(),
            (int*) &rSize[0], gPrevious, curData, FFTW_FORWARD, FFTW_ESTIMATE );

        /* copy original image into fftw_complex array and add random phase */
        #pragma omp parallel for
        for ( unsigned int i = 0; i < nElements; ++i )
        {
            curData[i][0] = rIntensity[i]; /* Re */
            curData[i][1] = 0;
        }
        fftwf_execute( toRealSpace );
        /* fftShift is not necessary, but I introduced this, because for the
         * example it shifted the result to a better looking position ... */
        //fftShift( curData, Nx,Ny );

        /* in the first step the last value for g is to be approximated
         * by g'. The last value for g, called g_k is needed, because
         * g_{k+1} = g_k - hioBeta * g' ! This is inside the loop
         * because the fft is needed */
        #pragma omp parallel for
        for ( auto i = 0u; i < nElements; ++i )
        {
            gPrevious[i][0] = curData[i][0]; // logf( 1 + fabs( curData[i][0] ) );
            gPrevious[i][1] = curData[i][1]; // logf( 1 + fabs( curData[i][1] ) );
        }

        /* repeatedly call HIO algorithm and change mask */
        for ( unsigned iCycleShrinkWrap = 0; iCycleShrinkWrap < rnCycles; ++iCycleShrinkWrap )
        {
            /************************** Update Mask ***************************/

            /* for iCycleShrinkWrap == 0 what actually is being calculated is
             * an object guess from the autocorrelation (fourier transform
             * of the intensity @see
             * https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem */

            /* blur |g'| (normally g' should be real!, so |.| not necessary) */
            complexNormElementwise( isMasked, curData, nElements );
            /* quick fix for the problem that larger images result in a larger
             * sum therefore a very largue 0-frequency value which disturbs the
             * mask threshold cut-off, see issue #40 A better solution is
             * necessary */
            //if ( iCycleShrinkWrap == 0 )
            //{
            //    isMasked[0] = 0;
            //    isMasked[Nx-1] = 0;
            //    isMasked[Nx*Ny-1] = 0;
            //    isMasked[Nx*Ny-(Ny-1)] = 0;
            //}
            //#pragma omp parallel for
            //for ( auto i = 0u; i < nElements; ++i )
            //    isMasked[i] = logf( isMasked[i] );
            libs::gaussianBlur( isMasked, Nx, Ny, sigma );

            #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
            {
                std::stringstream fname;
                fname << "shrinkWrap_iC-" << iCycleShrinkWrap
                      << "-mask-blurred.png";
                imresh::io::writeOutFuncs::writeOutPNG(
                    isMasked,
                    std::pair< unsigned int, unsigned int >{ Nx, Ny },
                    fname.str().c_str()
                );
            }
            {
                std::stringstream fname;
                fname << "shrinkWrap_iC-" << iCycleShrinkWrap
                      << "-mask-blurred-log-scale.png";

                auto logMask = new float[nElements];
                #pragma omp parallel for
                for ( auto i = 0u; i < nElements; ++i )
                    logMask[i] = logf( isMasked[i] );

                imresh::io::writeOutFuncs::writeOutAndFreePNG(
                    logMask,
                    std::pair< unsigned int, unsigned int >{ Nx, Ny },
                    fname.str().c_str()
                );
            }
            #endif

            /* apply threshold to make binary mask */
            const auto absMax = vectorMax( isMasked, nElements );
            float threshold = absMax;
            if ( iCycleShrinkWrap == 0 )
                threshold *= rIntensityCutOffAutoCorel;
            else
                threshold *= rIntensityCutOff;
            #pragma omp parallel for
            for ( auto i = 0u; i < nElements; ++i )
                isMasked[i] = isMasked[i] < threshold ? 1 : 0;

            /* update the blurring sigma */
            sigma = fmax( 1.5, ( 1 - rSigmaChange ) * sigma );

            #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
            {
                std::stringstream fname;
                fname << "shrinkWrap_iC-" << iCycleShrinkWrap
                      << "-mask.png";
                imresh::io::writeOutFuncs::writeOutPNG(
                    isMasked,
                    std::pair< unsigned int, unsigned int >{ Nx, Ny },
                    fname.str().c_str()
                );
            }
            #endif

            /*************** Run Hybrid-Input-Output algorithm ****************/

            for ( unsigned int iHioCycle = 0; iHioCycle < rnHioCycles; ++iHioCycle )
            {
                /* apply domain constraints to g' to get g */
                #pragma omp parallel for
                for ( unsigned int i = 0; i < nElements; ++i )
                {
                    if ( isMasked[i] == 1 or /* g' */ curData[i][0] < 0 )
                    {
                        gPrevious[i][0] -= rHioBeta * curData[i][0];
                        gPrevious[i][1] -= rHioBeta * curData[i][1];
                    }
                    else
                    {
                        gPrevious[i][0] = curData[i][0];
                        gPrevious[i][1] = curData[i][1];
                    }
                }

                /* Transform new guess g for f back into frequency space G' */
                fftwf_execute( toFreqSpace );

                /* Replace absolute of G' with measured absolute |F| */
                applyComplexModulus( curData, rIntensity, nElements );

                #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
                {
                    std::stringstream fname;
                    fname << "shrinkWrap_iC-" << iCycleShrinkWrap
                          << "_iHio-" << iHioCycle << "-intensity.png";

                    auto curIntensity = new float[nElements];
                    #pragma omp parallel for
                    for ( auto i = 0u; i < nElements; ++i )
                        curIntensity[i] = sqrtf( curData[i][0]*curData[i][0] + curData[i][1]*curData[i][1] );

                    imresh::io::writeOutFuncs::writeOutPNG(
                        curIntensity,
                        std::pair< unsigned int, unsigned int >{ Nx, Ny },
                        fname.str().c_str()
                    );
                }
                #endif

                fftwf_execute( toRealSpace );

                #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
                {
                    std::stringstream fname;
                    fname << "shrinkWrap_iC-" << iCycleShrinkWrap
                          << "_iHio-" << iHioCycle << "-object.png";

                    auto curObject = new float[nElements];
                    #pragma omp parallel for
                    for ( auto i = 0u; i < nElements; ++i )
                        curObject[i] = curData[i][0];

                    imresh::io::writeOutFuncs::writeOutPNG(
                        curObject,
                        std::pair< unsigned int, unsigned int >{ Nx, Ny },
                        fname.str().c_str()
                    );
                }
                #endif
            } // HIO loop

            /* check if we are done */
            const float currentError = imresh::libs::calculateHioError( curData /*g'*/, isMasked, nElements );
            #ifdef IMRESH_DEBUG
                std::cerr << "[Error " << currentError << "/" << rTargetError << "] "
                          << "[Cycle " << iCycleShrinkWrap << "/" << rnCycles-1 << "]"
                          << "\n";
            #endif
            if ( rTargetError > 0 && currentError < rTargetError )
                break;
            if ( iCycleShrinkWrap >= rnCycles )
                break;

            #ifdef WRITE_OUT_SHRINKWRAP_DEBUG
            {
                auto const tmpData = new float[nElements];
                #pragma omp parallel for
                for ( unsigned int i = 0; i < nElements; ++i )
                    tmpData[i] = curData[i][0];
                std::stringstream fname;
                fname << "shrinkWrap_iC-" << iCycleShrinkWrap
                      << "-object.png";
                imresh::io::writeOutFuncs::writeOutPNG(
                    tmpData,
                    std::pair< unsigned int, unsigned int >{ Nx, Ny },
                    fname.str().c_str()
                );
            }
            #endif
        } // shrink wrap loop

        #pragma omp parallel for
        for ( unsigned int i = 0; i < nElements; ++i )
            rIntensity[i] = curData[i][0];

        /* free buffers and plans */
        fftwf_destroy_plan( toFreqSpace );
        fftwf_destroy_plan( toRealSpace );
        fftwf_free( curData  );
        fftwf_free( gPrevious);
        delete[] isMasked;

        return 0;
    }


    /* explicit template instantiations */

    template
    void fftShift<fftwf_complex>
    (
        fftwf_complex * const & data,
        const unsigned & Nx,
        const unsigned & Ny
    );
#endif

    template
    void fftShift<float>
    (
        float * const & data,
        const unsigned & Nx,
        const unsigned & Ny
    );


} // namespace algorithms
} // namespace imresh

#ifdef DEBUG_SHRINKWRAPP_CPP
#   undef DEBUG_SHRINKWRAPP_CPP
#endif
#ifdef WRITE_OUT_SHRINKWRAP_DEBUG
#   undef WRITE_OUT_SHRINKWRAP_DEBUG
#endif
