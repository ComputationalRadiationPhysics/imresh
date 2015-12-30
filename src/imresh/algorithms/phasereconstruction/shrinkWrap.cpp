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


#include "shrinkWrap.h"


namespace imresh
{
namespace algorithms
{
namespace phasereconstruction
{


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
    int shrinkWrap
    (
        float * const & rIntensity,
        const std::vector<unsigned> & rSize,
        float rnHioCycles,
        float rTargetError,
        float rHioBeta,
        float rIntensityCutOffAutoCorel,
        float rIntensityCutOff,
        float rSigma0,
        float rSigmaChange,
        unsigned rnCycles,
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
        if ( rnHioCycles               <= 0 ) rnHioCycles               = 20;
        if ( rHioBeta                  <= 0 ) rHioBeta                  = 0.9;
        if ( rIntensityCutOffAutoCorel <= 0 ) rIntensityCutOffAutoCorel = 0.04;
        if ( rIntensityCutOff          <= 0 ) rIntensityCutOff          = 0.2;
        if ( rSigma0                   <= 0 ) rSigma0                   = 3.0;
        if ( rSigmaChange              <= 0 ) rSigmaChange              = 0.01;

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

        /* copy original image into fftw_complex array and add random phase */
        #pragma omp parallel for
        for ( unsigned i = 0; i < nElements; ++i )
        {
            curData[i][0] = rIntensity[i]; /* Re */
            curData[i][1] = 0;
        }

        /* create fft plans G' to g' and g to G */
        fftwf_plan toRealSpace = fftwf_plan_dft( rSize.size(),
            (int*) &rSize[0], curData, curData, FFTW_BACKWARD, FFTW_ESTIMATE );
        fftwf_plan toFreqSpace = fftwf_plan_dft( rSize.size(),
            (int*) &rSize[0], gPrevious, curData, FFTW_FORWARD, FFTW_ESTIMATE );

        float sigma = rSigma0;
        unsigned iCycle = 0;
        while(true)
        {
            /* In the first step use the autocorrelation (fourier transform of
             * the intensity @see
             * https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem */
            fftwf_execute( toRealSpace );

            /************************** Update Mask ***************************/
            std::cout << "Update Mask with sigma=" << sigma << "\n";

            /* blur |g'| (normally g' should be real!, so |.| not
             * necessary) */
            #pragma omp parallel for
            for ( unsigned i = 0; i < nElements; ++i )
            {
                /* in the inital step the complex norm cancels out the added
                 * random phase. This means the mask will be made from the
                 * input intensity in the initial run */
                const float & re = curData[i][0]; /* Re */
                const float & im = curData[i][1]; /* Im */
                isMasked[i] = sqrtf( re*re + im*im );

                /* in the first step the last value for g is to be approximated
                 * by g'. The last value for g, called g_k is needed, because
                 * g_{k+1} = g_k - hioBeta * g' ! */
                if ( iCycle == 0 )
                {
                    gPrevious[i][0] = re;
                    gPrevious[i][1] = im;
                }
            }
            gaussianBlur( isMasked, Nx, Ny, sigma );

            /* find maximum in order to calc threshold value */
            float absMax = 0;
            #pragma omp parallel for reduction( max : absMax )
            for ( unsigned i = 0; i < nElements; ++i )
                absMax = fmax( absMax, isMasked[i] );

            /* apply threshold to make binary mask */
            const float threshold = rIntensityCutOff * absMax;
            #pragma omp parallel for
            for ( unsigned i = 0; i < nElements; ++i )
                isMasked[i] = isMasked[i] < threshold ? 1 : 0;

            /* update the blurring sigma */
            sigma = fmax( 0.5, ( 1 - rSigmaChange ) * sigma );
            /******************************************************************/

            /* check if we are done */
            if ( rTargetError > 0 &&
                 calculateHioError( curData /*g'*/, isMasked, nElements )
                 < rTargetError )
                break;
            if ( iCycle >= 1/*rnCycles */ )
                break;

            /* apply domain constraints to g' to get g */
            #pragma omp parallel for
            for ( unsigned i = 0; i < Nx*Ny; ++i )
            {
                if ( isMasked[i] == 1 or /* g' */ curData[i][0] < 0 )
                {
                    gPrevious[i][0] -= rHioBeta * curData[i][0];
                    gPrevious[i][1] -= rHioBeta * curData[i][1];
                }
            }

            /* Transform new guess g for f back into frequency space G' */
            fftwf_execute( toFreqSpace );

            /* Replace absolute of G' with measured absolute |F|, keep phase */
            #pragma omp parallel for
            for ( unsigned i = 0; i < Nx*Ny; ++i )
            {
                const auto & re = curData[i][0];
                const auto & im = curData[i][1];
                const float factor = rIntensity[i] / sqrtf(re*re+im*im);
                curData[i][0] *= factor;
                curData[i][1] *= factor;
            }

            ++iCycle;
        }
        memcpy( rIntensity, isMasked, sizeof(rIntensity[0])*nElements );
        rIntensity[0] = 1;

        /* free buffers and plans */
        fftwf_destroy_plan( toFreqSpace );
        fftwf_destroy_plan( toRealSpace );
        fftwf_free( curData  );
        fftwf_free( gPrevious);
        delete[] isMasked;

        return 0;
    }


} // namespace phasereconstruction
} // namespace algorithms
} // namespace imresh
