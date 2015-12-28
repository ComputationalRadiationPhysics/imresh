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


int hybridInputOutput
(
    float * const rIoData,
    const uint8_t * const rMask,
    const unsigned rNx,
    const unsigned rNy,
    int rnCycles = INT_MAX,
    float rBeta = 0.9,
    unsigned rnCores = 0
)
{
    /* Evaluate input parameters and fill with default values if necessary */
    if ( rIoData == NULL or rMask == NULL ) return 1;
    if ( rNx == 0 or rNy == 0 ) return 1;
    if ( rnCycles < 0 ) rnCycles = INT_MAX;
    if ( rBeta    < 0 ) rBeta    = 0.9;
    if ( rnCores == 0 )
        rnCores = omp_get_num_threads();
    else
        omp_set_num_threads( rnCores );

    /* allocate arrays needed */
    fftw_complex * curData = fftw_alloc_complex( Nx*Ny );

    /* copy original intensity pattern to first array */
    for ( unsigned i = 0; i < Nx*Ny; ++i )
        mImageState[0][i] = fftw_complex{ rpOriginalData[i], 0 );

    using namespace imresh::math::image;
    using namespace imresh::math::vector;

    /* in the initial step introduce a random phase as a first guess */
#   if false
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
#    endif

        /* From here forth the periodic algorithm cycle steps begin!
         * (Actually the above 2 already belonged to the algorithm cycle
         * but those steps are different for the first rund than for
         * subsequent runs) */
        const int cycleOffset = 7;
        const int cycleFrame  = mCurrentFrame - cycleOffset;
        const int cyclePeriod = 4;

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

                /* blur IFT[ G'(x)**2 ] not g'=IFT[G'(x)] !! */
                fftw_complex * tmp = fftw_alloc_complex( Nx*Ny );
                #if false
                    /* square G' */
                    for ( unsigned i = 0; i < Nx*Ny; ++i )
                    {
                        const float & re = GPrime[i][0]; /* Re */
                        const float & im = GPrime[i][1]; /* Im */
                        tmp[i][0] = re*re + im*im; /* Re */
                        tmp[i][1] = 0;  /* Im */
                    }
                    /* IFT squared G' */
                    fftw_plan fft = fftw_plan_dft_2d( Nx,Ny, tmp, tmp,
                        FFTW_BACKWARD, FFTW_ESTIMATE );
                    fftw_execute(fft);
                    fftw_destroy_plan(fft);
                    /* shift real space variant, may not be necessary ... */
                    fftShift( tmp, Nx,Ny );
                #else
                    /* square g' */
                    for ( unsigned i = 0; i < Nx*Ny; ++i )
                    {
                        const float & re = gPrime[i][0]; /* Re */
                        const float & im = gPrime[i][1]; /* Im */
                        tmp[i][0] = re*re + im*im; /* Re */
                        tmp[i][1] = 0;  /* Im */
                    }
                #endif
                /* copy result into float array, because blur can't handle
                 * fftw_complex array */
                for ( unsigned i = 0; i < Nx*Ny; ++i )
                    mBlurred[i] = tmp[i][0];
                /* blur IFT[ G'(x)**2 ] */
                gaussianBlur( mBlurred, Nx, Ny, mSigma );

                /* make mask from autocorrelation */
                const float absMax = vectorMaxAbs( mBlurred, Nx*Ny );
                memcpy( mMask, mBlurred, Nx*Ny*sizeof(mMask[0]) );
                for ( unsigned i = 0; i < Nx*Ny; ++i )
                    mMask[i] = mMask[i] < intensityCutOff*absMax ? 0 : 1;

                mSigma = std::max( 0.5, (1-0.01)*mSigma );
                fftw_free(tmp);
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

        for ( unsigned i = 0; i < mnSteps; ++i )
            fftw_free( mImageState[i] );
        delete[] mMask;
        delete[] mBlurred;

}


} // namespace phasereconstruction
} // namespace algorithms
} // namespace imresh
