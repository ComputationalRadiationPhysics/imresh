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


#include "hybridInputOutput.h"


namespace imresh
{
namespace algorithms
{
namespace phasereconstruction
{


int hybridInputOutput
(
    float * const & rIoData,
    const uint8_t * const & rMask,
    const std::vector<unsigned> & rSize,
    unsigned rnCycles,
    float rTargetErr,
    float rBeta,
    unsigned rnCores
)
{
    if ( rSize.size() != 2 ) return 1;
    const unsigned & Ny = rSize[1];
    const unsigned & Nx = rSize[0];

    /* Evaluate input parameters and fill with default values if necessary */
    if ( rIoData == NULL or rMask == NULL ) return 1;
    if ( Nx == 0 or Ny == 0 ) return 1;
    if ( rnCycles == 0 ) rnCycles = UINT_MAX;
    if ( rBeta    < 0 ) rBeta    = 0.9;
    if ( rnCores == 0 )
        rnCores = omp_get_num_threads();
    else
        omp_set_num_threads( rnCores );

    /* allocate arrays needed */
    fftwf_complex * curData   = fftwf_alloc_complex( Nx*Ny );
    fftwf_complex * gPrevious = fftwf_alloc_complex( Nx*Ny );

    /* copy original intensity pattern to first array */
    for ( unsigned i = 0; i < Nx*Ny; ++i )
    {
        curData[i][0] = rIoData[i];
        curData[i][1] = 0;
    }

    /* create fft plans */
    fftwf_plan toRealSpace = fftwf_plan_dft_2d( Nx,Ny, curData, curData,
        FFTW_BACKWARD, FFTW_ESTIMATE );
    fftwf_plan toFreqSpace = fftwf_plan_dft_2d( Nx,Ny, curData, curData,
        FFTW_FORWARD, FFTW_ESTIMATE );

    /* in the initial step introduce a random phase as a first guess */
#   if true
        /* Because we constrain the object to be a real image (e.g. no
         * absorption which would result in an imaginary structure
         * coefficient), we should choose the random phases in such a way,
         * that the resulting fourier transformed will also be real */
        /* initialize a random real object */
        fftwf_complex * tmpRandReal = fftwf_alloc_complex( Nx*Ny );
        srand( 2623091912 );
        for ( unsigned i = 0; i < Nx*Ny; ++i )
        {
            tmpRandReal[i][0] = (float) rand() / RAND_MAX; /* Re */
            tmpRandReal[i][1] = 0; /* Im */
        }

        /* create and execute fftw plan */
        fftwf_plan planForward = fftwf_plan_dft_2d( Nx,Ny,
            tmpRandReal, tmpRandReal, FFTW_FORWARD, FFTW_ESTIMATE );
        fftwf_execute(planForward);
        fftwf_destroy_plan(planForward);

        /* applies phases of fourier transformed real random field to
         * measured input intensity */
        for ( unsigned i = 0; i < Nx*Ny; ++i )
        {
            /* get phase */
            const std::complex<float> z( tmpRandReal[i][0], tmpRandReal[i][1] );
            const float phase = std::arg( z );
            /* apply phase */
            curData[i][0] = rIoData[i] * cos(phase); /* Re */
            curData[i][1] = rIoData[i] * sin(phase); /* Im */
        }
        fftwf_free( tmpRandReal );
#    endif


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
        if ( rTargetErr > 0.0f )
        {
            float avgAbsDiff = 0;
            unsigned nSummands = 0;
            #pragma omp parallel for reduction( + : avgAbsDiff,nSummands )
            for ( unsigned i = 0; i < Nx*Ny; ++i )
            {
                /* g' */
                const auto & re = curData[i][0];
                const auto & im = curData[i][1];
                /* only add up norm where no object should be (rMask == 0) */
                avgAbsDiff += (1-rMask[i])*( re*re+im*im );
                nSummands  += (1-rMask[i]);
            }
            avgAbsDiff = sqrtf( avgAbsDiff ) / (float) nSummands;
            std::cout << "error = " << avgAbsDiff << "\n";
            if ( avgAbsDiff < rTargetErr )
                break;
        }
        if ( iCycle >= rnCycles )
            break;

        /* apply domain constraints to g' to get g */
        #pragma omp parallel for
        for ( unsigned i = 0; i < Nx*Ny; ++i )
        {
            if ( rMask[i] == 0 or /* g' */ curData[i][0] < 0 )
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


} // namespace phasereconstruction
} // namespace algorithms
} // namespace imresh
