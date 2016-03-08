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


#include "hybridInputOutput.hpp"

#include <cstddef>    // NULL
#include <cstdint>    // uint8_t
#include <climits>    // INT_MAX
#include <cstring>    // memcpy
#include <cmath>      // sqrtf
#include <complex>
#include <cassert>
#include <cfloat>     // FLT_EPSILON
#include <iostream>
#include <vector>
#include <omp.h>      // omp_get_num_procs, omp_set_num_procs
#ifdef USE_FFTW
#   include <fftw3.h>
#endif
#include "libs/vectorIndex.hpp"


namespace imresh
{
namespace libs
{


    /**
     * "For the input-output algorithms the error @f[ E_F @f] is
     *  usually meaningless since the input @f[ g_k(x) @f] is no longer
     *  an estimate of the object. Then the meaningful error
     *  is the object-domain error @f[ E_0 @f] given by Eq. (15)."
     *                                      (Fienup82)
     * Eq.15:
     * @f[ E_{0k}^2 = \sum\limits_{x\in\gamma} |g_k'(x)^2|^2 @f]
     * where @f[ \gamma @f] is the domain at which the constraints are
     * not met. SO this is the sum over the domain which should
     * be 0.
     *
     * Eq.16:
     * @f[ E_{Fk}^2 = \sum\limits_{u} |G_k(u) - G_k'(u)|^2 / N^2
                    = \sum_x |g_k(x) - g_k'(x)|^2 @f]
     **/
    template< class T_COMPLEX, class T_MASK_ELEMENT >
    float calculateHioError
    (
        T_COMPLEX      const * const __restrict__ gPrime,
        T_MASK_ELEMENT const * const __restrict__ rIsMasked,
        unsigned int const nElements,
        bool const rInvertMask,
        float * const __restrict__ rpTotalError,
        float * const __restrict__ rpnMaskedPixels
    )
    {
        float totalError    = 0;
        float nMaskedPixels = 0;

        #pragma omp parallel for reduction( + : totalError, nMaskedPixels )
        for ( unsigned i = 0; i < nElements; ++i )
        {
            const auto & re = gPrime[i][0];
            const auto & im = gPrime[i][1];

            /* only add up norm where no object should be (rMask == 0) */
            assert( rIsMasked[i] >= 0.0 and rIsMasked[i] <= 1.0 );
            float shouldBeZero = rIsMasked[i];
            if ( rInvertMask )
                shouldBeZero = 1 - shouldBeZero;

            totalError    += shouldBeZero * sqrtf( re*re+im*im );
            nMaskedPixels += shouldBeZero;
        }

        if ( rpTotalError != NULL )
            *rpTotalError    = totalError;
        if ( rpnMaskedPixels != NULL )
            *rpnMaskedPixels = nMaskedPixels;

        return sqrtf( totalError ) / (float) nMaskedPixels;
    }

#ifdef USE_FFTW
    template float calculateHioError<fftwf_complex,float>
    (
        fftwf_complex const * const __restrict__ gPrime,
        float         const * const __restrict__ rIsMasked,
        unsigned const nElements,
        bool     const rInvertMask,
        float * const __restrict__ rpTotalError,
        float * const __restrict__ rpnMaskedPixels
    );
    template float calculateHioError<fftw_complex,float>
    (
        fftw_complex const * const __restrict__ gPrime,
        float        const * const __restrict__ rIsMasked,
        unsigned const nElements,
        bool     const rInvertMask,
        float * const __restrict__ rpTotalError,
        float * const __restrict__ rpnMaskedPixels
    );


    template< class T_PREC >
    void addRandomPhase
    (
        const T_PREC * const & rNorm,
        fftwf_complex * const & rOutput,
        const std::vector<unsigned> & rSize,
        const unsigned & rnElements
    )
    {
        /* In the initial step introduce a random phase as a first guess.
         * This could be done without the fourier transform and extra space
         * by applying the needed symmetry f(-x) = \overline{f(x)}, so that
         * the inverse fourier trafo will be real.
         * Initially this condition should be true, because the input is
         * expected to stem from a fourier transform of a real function.
         * Note that the input is expected to have x=0 at ix=0, which means
         * we need to check wheter rIoData[ix==1] == rIoData[ix==Nx-1]
         */
#       ifndef NDEBUG
            for ( unsigned i = 0; i < rnElements; ++i )
            {
                auto iVec = convertLinearToVectorIndex( i, rSize );
                bool firstElementInOneDim = false;
                for ( unsigned iDim = 0; iDim < rSize.size(); ++iDim )
                {
                    firstElementInOneDim |= iVec[iDim] == 0;
                    iVec[iDim] = rSize[iDim] - iVec[iDim];
                }
                if ( firstElementInOneDim )
                    continue;
                unsigned iMirrored = convertVectorToLinearIndex( iVec, rSize );

                float max = fmax( fabs( rNorm[i] ), fabs( rNorm[iMirrored] ) );
                if ( max == 0 )
                    continue;
                //std::cout << "iy = " << iy << ": " << fabs( rIoData[iy*Nx + ix] - rIoData[(Ny-iy)*Nx + ix] ) / max << "\n";
                assert( fabs( rNorm[i] - rNorm[iMirrored] ) / max
                        < 100*FLT_EPSILON );
            }
#       endif

        /* Because we constrain the object to be a real image (e.g. no
         * absorption which would result in an imaginary structure
         * coefficient), we should choose the random phases in such a way,
         * that the resulting fourier transformed will also be real */
        /* initialize a random real object */
        fftwf_complex * tmpRandReal = fftwf_alloc_complex( rnElements );
        srand( 2623091912 );
        for ( unsigned i = 0; i < rnElements; ++i )
        {
            tmpRandReal[i][0] = (float) rand() / RAND_MAX; /* Re */
            tmpRandReal[i][1] = 0; /* Im */
        }

        /* create and execute fftw plan */
        fftwf_plan planForward = fftwf_plan_dft( rSize.size(), (int*) &rSize[0],
            tmpRandReal, tmpRandReal, FFTW_FORWARD, FFTW_ESTIMATE );
        fftwf_execute(planForward);
        fftwf_destroy_plan(planForward);

        /* applies phases of fourier transformed real random field to
         * measured input intensity */
        for ( unsigned i = 0; i < rnElements; ++i )
        {
            /* get phase */
            const std::complex<float> z( tmpRandReal[i][0], tmpRandReal[i][1] );
            const float phase = std::arg( z );
            /* apply phase */
            rOutput[i][0] = rNorm[i] * cos(phase); /* Re */
            rOutput[i][1] = rNorm[i] * sin(phase); /* Im */
        }
        fftwf_free( tmpRandReal );
    }


    int hybridInputOutput
    (
        float * const rIoData,
        uint8_t const * const rIsMasked,
        std::vector<unsigned int> const rSize,
        unsigned int rnCycles,
        float rTargetError,
        float rBeta,
        unsigned int rnCores
    )
    {
        if ( rSize.size() != 2 ) return 1;
        const unsigned & Ny = rSize[1];
        const unsigned & Nx = rSize[0];

        unsigned nElements = 1;
        for ( unsigned i = 0; i < rSize.size(); ++i )
        {
            assert( rSize[i] > 0 );
            nElements *= rSize[i];
        }

        /* Evaluate input parameters and fill with default values if necessary */
        if ( rIoData == NULL or rIsMasked == NULL ) return 1;
        if ( rnCycles == 0 ) rnCycles = UINT_MAX;
        if ( rBeta    <= 0 ) rBeta = 0.9;
        if ( rnCores  == 0 )
            rnCores = omp_get_num_threads();
        else
            omp_set_num_threads( rnCores );

        /* allocate arrays needed */
        fftwf_complex * curData   = fftwf_alloc_complex( Nx*Ny );
        fftwf_complex * gPrevious = fftwf_alloc_complex( Nx*Ny );

        /* create fft plans G' to g' and g to G */
        fftwf_plan toRealSpace = fftwf_plan_dft( rSize.size(),
            (int*) &rSize[0], curData, curData, FFTW_BACKWARD, FFTW_ESTIMATE );
        fftwf_plan toFreqSpace = fftwf_plan_dft( rSize.size(),
            (int*) &rSize[0], curData, curData, FFTW_FORWARD, FFTW_ESTIMATE );

        /* copy intensity and add random phase */
        addRandomPhase( rIoData, curData, rSize, nElements );


        unsigned iCycle = 0;
        while (true)
        {
            /* G' -> g' */
            fftwf_execute( toRealSpace );

            /* in the first step the last value for g is to be approximated
             * by g'. The last value for g, called g_k is needed, because
             * g_{k+1} = g_k - hioBeta * g' ! */
            if ( iCycle == 0 )
                memcpy( gPrevious, curData, Nx*Ny*sizeof( curData[0] ) );

            /* check if we are done */
            if ( rTargetError > 0 &&
                 calculateHioError( curData /*g'*/, rIsMasked, nElements )
                 < rTargetError )
                break;
            if ( iCycle >= rnCycles )
                break;

            /* apply domain constraints to g' to get g */
            #pragma omp parallel for
            for ( unsigned i = 0; i < Nx*Ny; ++i )
            {
                if ( rIsMasked[i] == 1 or /* g' */ curData[i][0] < 0 )
                {
                    curData[i][0] = gPrevious[i][0] - rBeta * curData[i][0];
                    curData[i][1] = gPrevious[i][1] - rBeta * curData[i][1];
                }
            }
            memcpy( gPrevious, curData, Nx*Ny*sizeof( curData[0] ) );

            /* Transform new guess g for f back into frequency space G' */
            fftwf_execute( toFreqSpace );

            /* Replace absolute of G' with measured absolute |F|, keep phase */
            #pragma omp parallel for
            for ( unsigned i = 0; i < Nx*Ny; ++i )
            {
                const auto & re = curData[i][0];
                const auto & im = curData[i][1];
                const float factor = rIoData[i] / sqrtf(re*re+im*im);
                curData[i][0] *= factor;
                curData[i][1] *= factor;
            }

            ++iCycle;
        }
        /* copy result back to output */
        #pragma omp parallel for
        for ( unsigned i = 0; i < Nx*Ny; ++i )
            rIoData[i] = curData[i][0];

        /* free buffers and plans */
        fftwf_destroy_plan( toFreqSpace );
        fftwf_destroy_plan( toRealSpace );
        fftwf_free( curData );
        fftwf_free( gPrevious );

        return 0; // success
    }
#endif


} // namespace libs
} // namespace imresh
