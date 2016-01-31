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


#define DEBUG_SHRINKWRAPP_CPP 1

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
#ifdef USE_PNG
#   if DEBUG_SHRINKWRAPP_CPP == 1
#       include "io/writeOutFuncs/writeOutFuncs.hpp"
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
        float * const & rIntensity,
        const std::vector<unsigned> & rSize,
        unsigned rnCycles,
        float rTargetError,
        float rHioBeta,
        float rIntensityCutOffAutoCorel,
        float rIntensityCutOff,
        float rSigma0,
        float rSigmaChange,
        unsigned rnHioCycles
    )
    {
        if ( rSize.size() != 2 ) return 1;
        const unsigned & Ny = rSize[1];
        const unsigned & Nx = rSize[0];

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
        fftwf_complex * const curData   = fftwf_alloc_complex( nElements );
        fftwf_complex * const gPrevious = fftwf_alloc_complex( nElements );
        auto const isMasked = new float[nElements];

        /* create fft plans G' to g' and g to G */
        auto toRealSpace = fftwf_plan_dft( rSize.size(),
            (int*) &rSize[0], curData, curData, FFTW_BACKWARD, FFTW_ESTIMATE );
        auto toFreqSpace = fftwf_plan_dft( rSize.size(),
            (int*) &rSize[0], gPrevious, curData, FFTW_FORWARD, FFTW_ESTIMATE );

        /* create first guess for mask from autocorrelation (fourier transform
         * of the intensity @see
         * https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem */
        #pragma omp parallel for
        for ( unsigned i = 0; i < nElements; ++i )
        {
            curData[i][0] = rIntensity[i]; /* Re */
            curData[i][1] = 0;
        }
        fftwf_execute( toRealSpace );
        complexNormElementwise( isMasked, curData, nElements );
        /* fftShift is not necessary, but I introduced this, because for the
         * example it shifted the result to a better looking position ... */
        //fftShift( isMasked, Nx,Ny );
        libs::gaussianBlur( isMasked, Nx, Ny, sigma );

        #ifdef USE_PNG
        #if DEBUG_SHRINKWRAPP_CPP == 1
            imresh::io::writeOutFuncs::writeOutPNG( isMasked, std::pair<unsigned,unsigned>{Nx,Ny}, "shrinkWrap-init-mask-blurred.png" );
        #endif
        #endif

        /* apply threshold to make binary mask */
        {
            const auto absMax = vectorMax( isMasked, nElements );
            const float threshold = rIntensityCutOffAutoCorel * absMax;
            #pragma omp parallel for
            for ( unsigned i = 0; i < nElements; ++i )
                isMasked[i] = isMasked[i] < threshold ? 1 : 0;
        }

        #ifdef USE_PNG
        #if DEBUG_SHRINKWRAPP_CPP == 1
            imresh::io::writeOutFuncs::writeOutPNG( isMasked, std::pair<unsigned,unsigned>{Nx,Ny}, "shrinkWrap-init-mask.png" );
        #endif
        #endif

        /* copy original image into fftw_complex array and add random phase */
        #pragma omp parallel for
        for ( unsigned i = 0; i < nElements; ++i )
        {
            curData[i][0] = rIntensity[i]; /* Re */
            curData[i][1] = 0;
        }

        /* in the first step the last value for g is to be approximated
         * by g'. The last value for g, called g_k is needed, because
         * g_{k+1} = g_k - hioBeta * g' ! This is inside the loop
         * because the fft is needed */
        #pragma omp parallel for
        for ( unsigned i = 0; i < nElements; ++i )
        {
            gPrevious[i][0] = curData[i][0];
            gPrevious[i][1] = curData[i][1];
        }

        /* repeatedly call HIO algorithm and change mask */
        for ( unsigned iCycleShrinkWrap = 0; iCycleShrinkWrap < rnCycles; ++iCycleShrinkWrap )
        {
            /************************** Update Mask ***************************/
            std::cout << "Update Mask with sigma=" << sigma << "\n";

            /* blur |g'| (normally g' should be real!, so |.| not necessary) */
            complexNormElementwise( isMasked, curData, nElements );
            libs::gaussianBlur( isMasked, Nx, Ny, sigma );
            const auto absMax = vectorMax( isMasked, nElements );
            /* apply threshold to make binary mask */
            const float threshold = rIntensityCutOff * absMax;
            #pragma omp parallel for
            for ( unsigned i = 0; i < nElements; ++i )
                isMasked[i] = isMasked[i] < threshold ? 1 : 0;

            /* update the blurring sigma */
            sigma = fmax( 1.5, ( 1 - rSigmaChange ) * sigma );

            for ( unsigned iHioCycle = 0; iHioCycle < rnHioCycles; ++iHioCycle )
            {
                /* apply domain constraints to g' to get g */
                #pragma omp parallel for
                for ( unsigned i = 0; i < nElements; ++i )
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

                fftwf_execute( toRealSpace );
            } // HIO loop

            /* check if we are done */
            const float currentError = imresh::libs::calculateHioError( curData /*g'*/, isMasked, nElements );
            std::cout << "[Error " << currentError << "/" << rTargetError << "] "
                      << "[Cycle " << iCycleShrinkWrap << "/" << rnCycles-1 << "]"
                      << "\n";
            if ( rTargetError > 0 && currentError < rTargetError )
                break;
            if ( iCycleShrinkWrap >= rnCycles )
                break;
        } // shrink wrap loop
        for ( unsigned i = 0; i < nElements; ++i )
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
