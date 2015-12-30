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
        /* needed variables */
        int mCurrentFrame;

        const unsigned Nx = rSize[1];
        const unsigned Ny = rSize[0];
        static constexpr unsigned mnSteps = 8;
        fftw_complex * mImageState[ mnSteps ];
        std::string mTitles[ mnSteps+3 ];
        std::vector<float> mReconstructedErrors;

        float * mBlurred, * mMask;

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
            mImageState[i] = fftw_alloc_complex(Nx*Ny);
            memset( mImageState[i], 0, sizeof(mImageState[i][0])*Nx*Ny );
        }
        mBlurred = new float[ Nx*Ny ];
        mMask    = new float[ Nx*Ny ];
        memset( mBlurred, 0, Nx*Ny*sizeof(mBlurred[0]) );
        memset( mMask   , 0, Nx*Ny*sizeof(mMask   [0]) );

        /* save original image to first array */
        for ( unsigned i = 0; i < Nx*Ny; ++i )
            mImageState[2][i][0] = rIntensity[i]; /* Re */


        using namespace imresh::algorithms;

        /* define some aliases according to Fienup82 */
        const auto & absF     = mImageState[2];
        const auto & autocorr = mImageState[3]; // autocorrelation
        const auto & GPrime   = mImageState[4];
        const auto & gPrime   = mImageState[5];
        const auto & g        = mImageState[6];
        const auto & G        = mImageState[7];

        const int cycleOffset = 7;
        const int cyclePeriod = 4;
for ( unsigned int iCycle = 0; iCycle < cycleOffset+50*cyclePeriod; ++iCycle )
{
        ++mCurrentFrame;
        if ( mCurrentFrame == 3 )
        {
            /* create and execute fftw plan */
            fftw_plan fft = fftw_plan_dft_2d( Nx,Ny, absF, autocorr,
                FFTW_BACKWARD, FFTW_ESTIMATE );
            fftw_execute(fft);
            fftw_destroy_plan(fft);
            fftShift( autocorr, Nx,Ny );
        }
        else if ( mCurrentFrame == 4 )
        {
            /* blur the autocorrelation function */
            for ( unsigned i = 0; i < Nx*Ny; ++i )
                mBlurred[i] = autocorr[i][0] > 0 ? autocorr[i][0] : 0;
            gaussianBlur( mBlurred, Nx, Ny, mSigma /*sigma*/ );
        }
        else if ( mCurrentFrame == 5 )
        {
            /* make mask from autocorrelation */
            float absMax = 0;
            #pragma omp parallel for reduction( max : absMax )
            for ( unsigned i = 0; i < Nx*Ny; ++i )
                absMax = fmax( absMax, fabs( mBlurred[i] ) );
            memcpy( mMask, mBlurred, Nx*Ny*sizeof(mMask[0]) );
            for ( unsigned i = 0; i < Nx*Ny; ++i )
                mMask[i] = mMask[i] < intensityCutOffAutoCorel*absMax ? 0 : 1;
        }
        else if ( mCurrentFrame == 6 )
        {
            /* in the initial step introduce a random phase as a first guess */
            memcpy( GPrime, absF, Nx*Ny*sizeof(absF[0]) );
#           if false
                /* Because we constrain the object to be a real image (e.g. no
                 * absorption which would result in an imaginary structure
                 * coefficient), we should choose the random phases in such a way,
                 * that the resulting fourier transformed will also be real */
                /* initialize a random real object */
                fftw_complex * tmpRandReal = fftw_alloc_complex( Nx*Ny );
                srand( 2623091912 );
                for ( unsigned i = 0; i < Nx*Ny; ++i )
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
                for ( unsigned i = 0; i < Nx*Ny; ++i )
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
        }

        /* From here forth the periodic algorithm cycle steps begin!
         * (Actually the above 2 already belonged to the algorithm cycle
         * but those steps are different for the first rund than for
         * subsequent runs) */
        const int cycleFrame  = mCurrentFrame - cycleOffset;

        if ( mCurrentFrame < 7 ) {}
        else if ( cycleFrame % cyclePeriod == 0 )
        {
            /* Transform G' into real space g' */
            fftw_plan fft = fftw_plan_dft_2d( Nx,Ny, GPrime, gPrime,
                FFTW_BACKWARD, FFTW_ESTIMATE );
            fftw_execute(fft);
            fftw_destroy_plan(fft);

            /* check if result is real! */
            float avgRe = 0, avgIm = 0;
            for ( unsigned i = 0; i < Nx*Ny; ++i )
            {
                avgRe += fabs( gPrime[i][0] );
                avgIm += fabs( gPrime[i][1] );
            }
            avgRe /= (float) Nx*Ny;
            avgIm /= (float) Nx*Ny;
            //std::cout << std::scientific
            //          << "Avg. Re = " << avgRe << "\n"
            //          << "Avg. Im = " << avgIm << "\n";
            //assert( avgIm < avgRe * 1e-5 );

            /* in the first step the last value for g is to be approximated
             * by g'. The last value for g, called g_k is needed, because
             * g_{k+1} = g_k - hioBeta * g' ! */
            if ( cycleFrame == 0 )
                memcpy( g, gPrime, sizeof(g[0])*Nx*Ny );
        }
        else if ( cycleFrame % cyclePeriod == 1 )
        {
            /* create a new mask (shrink-wrap) */
            if ( cycleFrame % ( nHioCycles*cyclePeriod ) == 1 && cycleFrame != 1 )
            {
                std::cout << "Update Mask with sigma="<<mSigma<<"\n";

                /* blur |g'| (normally g' should be real!, so |.| not
                 * necessary) */
                #pragma omp parallel for
                for ( unsigned i = 0; i < Nx*Ny; ++i )
                {
                    const float & re = gPrime[i][0]; /* Re */
                    const float & im = gPrime[i][1]; /* Im */
                    mBlurred[i] = sqrtf( re*re + im*im );
                }
                gaussianBlur( mBlurred, Nx, Ny, mSigma );

                float absMax = 0;
                #pragma omp parallel for reduction( max : absMax )
                for ( unsigned i = 0; i < Nx*Ny; ++i )
                    absMax = fmax( absMax, mBlurred[i] );

                /* make mask */
                memcpy( mMask, mBlurred, Nx*Ny*sizeof(mMask[0]) );
                for ( unsigned i = 0; i < Nx*Ny; ++i )
                    mMask[i] = mMask[i] < intensityCutOff*absMax ? 0 : 1;

                mSigma = std::max( 0.5, (1-0.01)*mSigma );
            }

            /* buffer domain gamma where g' does not satisfy object constraints */
            float * gamma = new float[Nx*Ny];
            for ( unsigned i = 0; i < Nx*Ny; ++i )
                //gamma[i] = ( mask[i][0] == 0 or gPrime[i][0] < 0 );
                gamma[i] = mMask[i] == 0 or gPrime[i][0] < 0 ? 1 : 0;

            /* apply domain constraints to g' to get g */
            for ( unsigned i = 0; i < Nx*Ny; ++i )
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
        }
        else if ( cycleFrame % cyclePeriod == 2 )
        {
            /* Transform new guess g for f back into frequency space G' */
            fftw_plan fft = fftw_plan_dft_2d( Nx,Ny, g, G,
                FFTW_FORWARD, FFTW_ESTIMATE );
            fftw_execute(fft);
            fftw_destroy_plan(fft);
        }
        else if ( cycleFrame % cyclePeriod == 3 )
        {
            /* Replace absolute of G' with measured absolute |F|, keep phase */
            for ( unsigned i = 0; i < Nx*Ny; ++i )
            {
                const auto & re = G[i][0];
                const auto & im = G[i][1];
                const float norm = sqrtf(re*re+im*im);
                GPrime[i][0] = absF[i][0] * G[i][0] / norm;
                GPrime[i][1] = absF[i][0] * G[i][1] / norm;
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
            float avgAbsDiff = 0;
            unsigned nSummands = 0;
            for ( unsigned i = 0; i < Nx*Ny; ++i )
            {
                const auto & re = gPrime[i][0];
                const auto & im = gPrime[i][1];
                if ( mMask[i] == 0 )
                {
                    avgAbsDiff += re*re+im*im;
                    nSummands += 1;
                }
            }
            /* delete enclosing log(...) if no log-plot wanted */
            mReconstructedErrors.push_back( log( sqrtf( avgAbsDiff ) / nSummands ) );
            //std::cout << "Current error = " << mReconstructedErrors.back() << "\n";
        }
}

        for ( unsigned i = 0; i < Nx*Ny; ++i )
            rIntensity[i] = fabs( gPrime[i][0] );

        return 0;
    }


} // namespace phasereconstruction
} // namespace algorithms
} // namespace imresh
