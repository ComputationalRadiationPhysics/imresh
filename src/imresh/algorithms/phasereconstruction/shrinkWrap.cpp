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




        /* needed variables */
        int mCurrentFrame = 0;

        static constexpr unsigned mnSteps = 8;
        fftw_complex * mImageState[ mnSteps ];
        std::string mTitles[ mnSteps+3 ];
        std::vector<float> mReconstructedErrors;

        float * mBlurred;

        typedef struct { int x0,y0,x1,y1; } Line2d;
        std::vector<Line2d> mArrows;

        static constexpr float hioBeta = 0.9;
        static constexpr float intensityCutOffAutoCorel = 0.04;
        static constexpr float intensityCutOff = 0.20;
        static constexpr float sigma0 = 3.0;
        static constexpr int   nHioCycles = 20;

        float mSigma = sigma0;

        /* allocate and clear all intermediary steps */
        for ( unsigned i = 0; i < mnSteps; ++i )
        {
            mImageState[i] = fftw_alloc_complex(nElements);
            memset( mImageState[i], 0, sizeof(mImageState[i][0])*nElements );
        }
        mBlurred = new float[ nElements ];
        memset( mBlurred, 0, nElements*sizeof(mBlurred[0]) );
        memset( isMasked   , 0, nElements*sizeof(isMasked   [0]) );

        /* save original image to first array */
        for ( unsigned i = 0; i < nElements; ++i )
            mImageState[2][i][0] = rIntensity[i]; /* Re */


        using namespace imresh::algorithms;

        /* define some aliases according to Fienup82 */
        const auto & absF     = mImageState[2];
        const auto & autocorr = mImageState[3]; // autocorrelation
        const auto & GPrime   = mImageState[4];
        const auto & gPrime   = mImageState[5];
        const auto & g        = mImageState[6];
        const auto & G        = mImageState[7];

        /* create and execute fftw plan */
        fftw_plan fft = fftw_plan_dft_2d( Nx,Ny, absF, autocorr,
            FFTW_BACKWARD, FFTW_ESTIMATE );
        fftw_execute(fft);
        fftw_destroy_plan(fft);
        fftShift( autocorr, Nx,Ny );

            /* blur the autocorrelation function */
            for ( unsigned i = 0; i < nElements; ++i )
                mBlurred[i] = autocorr[i][0] > 0 ? autocorr[i][0] : 0;
            gaussianBlur( mBlurred, Nx, Ny, mSigma /*sigma*/ );


            /* make mask from autocorrelation */
            float absMax = 0;
            #pragma omp parallel for reduction( max : absMax )
            for ( unsigned i = 0; i < nElements; ++i )
                absMax = fmax( absMax, fabs( mBlurred[i] ) );
            memcpy( isMasked, mBlurred, nElements*sizeof(isMasked[0]) );
            for ( unsigned i = 0; i < nElements; ++i )
                isMasked[i] = isMasked[i] < intensityCutOffAutoCorel*absMax ? 1 : 0;


            /* in the initial step introduce a random phase as a first guess */
            memcpy( GPrime, absF, nElements*sizeof(absF[0]) );
#           if false
                /* Because we constrain the object to be a real image (e.g. no
                 * absorption which would result in an imaginary structure
                 * coefficient), we should choose the random phases in such a way,
                 * that the resulting fourier transformed will also be real */
                /* initialize a random real object */
                fftw_complex * tmpRandReal = fftw_alloc_complex( nElements );
                srand( 2623091912 );
                for ( unsigned i = 0; i < nElements; ++i )
                {
                    tmpRandReal[i][0] = (float) rand() / RAND_MAX; /* Re */
                    tmpRandReal[i][1] = 0; /* Im */
                }

                /* create and execute fftw plan */
                fftw_plan planForward = fftw_plan_dft_2d( Nx,Ny,
                    tmpRandReal, tmpRandReal, FFTW_FORWARD, FFTW_ESTIMATE );
                fftw_execute(planForward);
                fftw_destroy_plan(planForward);

                /* applies phases of fourier transformed real random field to
                 * measured input intensity */
                for ( unsigned i = 0; i < nElements; ++i )
                {
                    /* get phase */
                    const std::complex<float> z( tmpRandReal[i][0], tmpRandReal[i][1] );
                    const float phase = std::arg( z );
                    /* apply phase */
                    const float & re = absF[i][0];
                    assert( absF[i][1] == 0 );
                    GPrime[i][0] = re * cos(phase); /* Re */
                    GPrime[i][1] = re * sin(phase); /* Im */
                }
                fftw_free( tmpRandReal );
#           endif


        const int cycleOffset = 7;
        const int cyclePeriod = 4;
        mCurrentFrame = -1;
for ( unsigned int iCycle = 0; iCycle < 100; ++iCycle )
{
        ++mCurrentFrame;

        /* From here forth the periodic algorithm cycle steps begin!
         * (Actually the above 2 already belonged to the algorithm cycle
         * but those steps are different for the first rund than for
         * subsequent runs) */

            /* Transform G' into real space g' */
            fftw_plan fft = fftw_plan_dft_2d( Nx,Ny, GPrime, gPrime,
                FFTW_BACKWARD, FFTW_ESTIMATE );
            fftw_execute(fft);
            fftw_destroy_plan(fft);

            /* check if result is real! */
            float avgRe = 0, avgIm = 0;
            for ( unsigned i = 0; i < nElements; ++i )
            {
                avgRe += fabs( gPrime[i][0] );
                avgIm += fabs( gPrime[i][1] );
            }
            avgRe /= (float) nElements;
            avgIm /= (float) nElements;
            //std::cout << std::scientific
            //          << "Avg. Re = " << avgRe << "\n"
            //          << "Avg. Im = " << avgIm << "\n";
            //assert( avgIm < avgRe * 1e-5 );

            /* in the first step the last value for g is to be approximated
             * by g'. The last value for g, called g_k is needed, because
             * g_{k+1} = g_k - hioBeta * g' ! */
            if ( mCurrentFrame == 0 )
                memcpy( g, gPrime, sizeof(g[0])*nElements );


            /* create a new mask (shrink-wrap) */
            if ( mCurrentFrame % nHioCycles == 0 && mCurrentFrame != 0 )
            {
                std::cout << "Update Mask with sigma="<<mSigma<<"\n";

                /* blur |g'| (normally g' should be real!, so |.| not
                 * necessary) */
                #pragma omp parallel for
                for ( unsigned i = 0; i < nElements; ++i )
                {
                    const float & re = gPrime[i][0]; /* Re */
                    const float & im = gPrime[i][1]; /* Im */
                    mBlurred[i] = sqrtf( re*re + im*im );
                }
                gaussianBlur( mBlurred, Nx, Ny, mSigma );

                float absMax = 0;
                #pragma omp parallel for reduction( max : absMax )
                for ( unsigned i = 0; i < nElements; ++i )
                    absMax = fmax( absMax, mBlurred[i] );

                /* make mask */
                memcpy( isMasked, mBlurred, nElements*sizeof(isMasked[0]) );
                for ( unsigned i = 0; i < nElements; ++i )
                    isMasked[i] = isMasked[i] < intensityCutOff*absMax ? 1 : 0;

                mSigma = std::max( 0.5, (1-0.01)*mSigma );
            }

            /* buffer domain gamma where g' does not satisfy object constraints */
            float * gamma = new float[nElements];
            for ( unsigned i = 0; i < nElements; ++i )
                //gamma[i] = ( mask[i][0] == 0 or gPrime[i][0] < 0 );
                gamma[i] = isMasked[i] == 1 or gPrime[i][0] < 0 ? 1 : 0;

            /* apply domain constraints to g' to get g */
            for ( unsigned i = 0; i < nElements; ++i )
            {
                if ( gamma[i] == 0 )
                {
                    g[i][0] = gPrime[i][0];
                    g[i][1] = gPrime[i][1]; /* shouldn't be necessary */
                }
                else
                {
                    g[i][0] -= hioBeta*gPrime[i][0];
                    g[i][1] -= hioBeta*gPrime[i][1]; /* shouldn't be necessary */
                }
            }

            delete[] gamma;


            /* Transform new guess g for f back into frequency space G' */
            fftw_plan fft2 = fftw_plan_dft_2d( Nx,Ny, g, G,
                FFTW_FORWARD, FFTW_ESTIMATE );
            fftw_execute( fft2 );
            fftw_destroy_plan( fft2 );


            /* Replace absolute of G' with measured absolute |F|, keep phase */
            for ( unsigned i = 0; i < nElements; ++i )
            {
                const auto & re = G[i][0];
                const auto & im = G[i][1];
                const float norm = sqrtf(re*re+im*im);
                GPrime[i][0] = rIntensity[i] * G[i][0] / norm;
                GPrime[i][1] = rIntensity[i] * G[i][1] / norm;
            }

            /* check if we are done */
            const float currentError = calculateHioError( gPrime /*g'*/,  isMasked, nElements, false /* invert mask */ );
            if ( rTargetError > 0 && currentError < rTargetError )
                break;
            //if ( iCycle >= 10/*rnCycles */ )
            //    break;

            std::cout << " =? " << log(currentError) << "\n";
}

        for ( unsigned i = 0; i < nElements; ++i )
            rIntensity[i] = fabs( gPrime[i][0] );

        return 0;
    }


} // namespace phasereconstruction
} // namespace algorithms
} // namespace imresh
