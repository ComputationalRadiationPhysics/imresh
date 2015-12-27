
#include <iostream>
#include <complex>
#include <cstdlib>
#include <cmath>
#include <string>
#include <algorithm>  // swap
#include <fftw3.h>
#include "sdl/sdlcommon.h"
#include "sdl/sdlplot.h"
#include "sdl/complexPlot.h"
#include "examples/createAtomCluster.cpp"
#include "examples/createSlit.cpp"
#include "math/vector/vectorReduce.h"
#include "math/image/gaussian.h"

#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif


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
void fftShift( fftw_complex * const data, const unsigned Nx, const unsigned Ny )
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
 *                error
 *                +---+         blurred         mask
 *                | A |          +---+         +---+
 *                +---+      +-> | 8 | ------> | 9 |
 *             +-------------+   +---+      /  +---+
 *             f        F          g'      /     g
 *           +---+ FT +---+      +---+   mask  +---+
 *           | 0 | -> | 1 |      | 5 | ------> | 6 |
 *           +---+    +---+      +---+         +---+
 *                  |.| |          ^             | FT
 *                      v       IFT|             v
 *           +---+IFT +---+ RAND +---+ use |F| +---+
 *           | 3 | <- | 2 | ---> | 4 | <------ | 7 |
 *           +---+    +---+      +---+    ^    +---+
 *                     |F|         G'     |      G
 *                      +-----------------+
 *
 * 0 ... original image f (normally we don't have that and want to
 *       reconstruct this)
 * 1 ... complex fourier transformed original image F
 * 2 ... The measured intesity of the difraction image, i.e. the absolute
 *       of the fourier transform in 1: |F|
 * 3 ... naive inverse fourier transform of 2 to show what the missing
 *       phase affects
 * 4 ... current guess G for F. In the first step this is |F|*exp(i*RANDOM)
 * 5 ...
 * A ... current difference to fourier transform of reconstructed function
 *       to measurement in order to observe convergence
 **/
class AnimateHybridInputOutput
{
private:
    int mCurrentFrame;

    const unsigned Nx, Ny;
    static constexpr unsigned mnSteps = 8;
    SDL_Rect mPlotPositions[ mnSteps+3 ]; /* +3: blurred, mask, error */
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

    float mSigma;

    /**
     * Draws an array to point from 1 rectangle to another, e.g. for plots
     *
     *        +----+    +----+        +----+    +----+
     * e.g.:  |from| -> | to |   or:  | to | <- |from|
     *        +----+    +----+        +----+    +----+
     *
     * @param[in] from
     * @param[in] to
     * @param[out] arrow will hold coordinates of arrow to draw
     **/
    void calcArrowFromRectToRect
    (
        SDL_Rect const * const from,
        SDL_Rect const * const to,
        Line2d * const arrow
    )
    {
        const int distx = abs( from->x - to->x );
        const int disty = abs( from->y - to->y );
        assert( distx + disty > 0 );
        if ( distx > disty )
        {
            arrow->y0 = from->y + from->h/2;
            arrow->y1 = to  ->y + to  ->h/2;
            if ( from->x < to->x ) /* -> */
            {
                arrow->x0 = from->x + 1.05*from->w;
                arrow->x1 = to  ->x - 0.15*from->w;
            }
            else /* <- */
            {
                arrow->x0 = from->x - 0.15*from->w;
                arrow->x1 = to  ->x + 1.05*from->w;
            }
        }
        else
        {
            arrow->x0 = from->x + from->w/2;
            arrow->x1 = to  ->x + to  ->w/2;
            if ( from->y < to->y ) /* v down */
            {
                arrow->y0 = from->y + 1.1*from->h;
                arrow->y1 = to  ->y - 0.1*from->h;
            }
            else /* ^ up */
            {
                arrow->y0 = from->y - 0.1*from->h;
                arrow->y1 = to  ->y + 1.1*from->h;
            }
        }
    }


public:
    AnimateHybridInputOutput
    (
      const float * const rpOriginalData,
      const unsigned rNx, const unsigned rNy
    )
     : mCurrentFrame(0), Nx(rNx), Ny(rNy), mSigma(sigma0)
    {
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
            mImageState[0][i][0] = rpOriginalData[i]; /* Re */

        /* scale very small images up to a height of at least 200 */
        const int multiplierX = (int) ceilf( 160.0f / (float) Nx );
        const int multiplierY = (int) ceilf( 160.0f / (float) Ny );
        const int plotWidth  = multiplierX * Nx;
        const int plotHeight = multiplierY * Ny;

        /* Initialize plot titles */
        mTitles[0] = "Original Image (f)";
        mTitles[1] = "FT[Original Image] (F)";
        mTitles[2] = "Diffraction Intensity (|F|)";
        mTitles[3] = "IFT[|F|]";
        mTitles[4] = "Current Guess for F (G')";
        mTitles[5] = "g'";
        mTitles[6] = "g";
        mTitles[7] = "G";
        mTitles[8] = "blurred f";
        mTitles[9] = "mask'";

        /* Initialize plot positions */
        {
        SDL_Rect tmp = { 40, 30 + int(1.3*plotHeight), plotWidth, plotHeight };
        mPlotPositions[0] = tmp;
        tmp.x += 1.3*plotWidth;     mPlotPositions[1] = tmp;
        tmp.y += 1.3*plotHeight;    mPlotPositions[2] = tmp;
        tmp.x -= 1.3*plotWidth;     mPlotPositions[3] = tmp;
        tmp.x += 3.0*plotWidth;     mPlotPositions[4] = tmp;
        tmp.y -= 1.3*plotHeight;    mPlotPositions[5] = tmp;
        tmp.x += 1.3*plotWidth;     mPlotPositions[6] = tmp;
        tmp.y += 1.3*plotHeight;    mPlotPositions[7] = tmp;
        tmp.x -= 1.3*plotWidth;
        tmp.y -= 2.6*plotHeight;    mPlotPositions[8] = tmp;
        tmp.x += 1.3*plotWidth;     mPlotPositions[9] = tmp;
        tmp.x = 40 + int(0.15*plotWidth);
        tmp.y = 30 + int(0.4 *plotHeight);
        tmp.w = 2.0*plotWidth;
        tmp.h = 0.5*plotWidth;
        mPlotPositions[10] = tmp;
        }

        /* Initialize arrows */
        {
        const auto & p = mPlotPositions;
        Line2d tmp;
        calcArrowFromRectToRect( p+0, p+1, &tmp ); mArrows.push_back( tmp );
        calcArrowFromRectToRect( p+1, p+2, &tmp ); mArrows.push_back( tmp );
        calcArrowFromRectToRect( p+2, p+3, &tmp ); mArrows.push_back( tmp );

        calcArrowFromRectToRect( p+2, p+4, &tmp ); mArrows.push_back( tmp );

        calcArrowFromRectToRect( p+4, p+5, &tmp ); mArrows.push_back( tmp );
        calcArrowFromRectToRect( p+5, p+6, &tmp ); mArrows.push_back( tmp );
        calcArrowFromRectToRect( p+6, p+7, &tmp ); mArrows.push_back( tmp );
        calcArrowFromRectToRect( p+7, p+4, &tmp ); mArrows.push_back( tmp );

        calcArrowFromRectToRect( p+5, p+8, &tmp ); mArrows.push_back( tmp );
        calcArrowFromRectToRect( p+8, p+9, &tmp ); mArrows.push_back( tmp );
        std::swap( tmp.x0, tmp.x1 );
        std::swap( tmp.y0, tmp.y1 );
        tmp.x1 = ( tmp.x0 + tmp.x1 ) / 2;
        tmp.y1 = p[5].y + p[5].h/2;
        tmp.y0 += 0.05 * p[5].h;
        tmp.y1 -= 0.1  * p[5].h;
        mArrows.push_back( tmp );
        }
    }

    ~AnimateHybridInputOutput()
    {
        for ( unsigned i = 0; i < mnSteps; ++i )
            fftw_free( mImageState[i] );
        delete[] mMask;
        delete[] mBlurred;
    }

    void render( SDL_Renderer * rpRenderer )
    {
        using namespace sdlcommon;

        SDL_SetRenderDrawColor( rpRenderer, 0,0,0,255 );
        /* Draw arrows */
        for ( const auto & l : mArrows )
            SDL_RenderDrawArrow( rpRenderer, l.x0, l.y0, l.x1, l.y1 );

        /* Draw complex plots */
        for ( unsigned i = 0; i < mnSteps; ++i )
        {
            bool isRealSpace = i == 0 /* original */
                            or i == 3 /* autocorelation */
                            or i == 5 /* g' */
                            or i == 6 /* g */;
            SDL_RenderDrawComplexMatrix( rpRenderer, mPlotPositions[i], 0,0,0,0,
                mImageState[i],Nx,Ny, true/*drawAxis*/, mTitles[i].c_str(),
                not isRealSpace/*log scale*/, not isRealSpace/*swapQuadrants*/,
                1 /* color map */ );
        }

        /* Draw blurred and mask plot */
        const float maxVal = imresh::math::vector::vectorMaxAbs( mBlurred, Nx*Ny);
        for ( unsigned i = 0; i < Nx*Ny; ++i )
            mBlurred[i] /= maxVal;
        SDL_RenderDrawMatrix( rpRenderer, mPlotPositions[8], 0,0,0,0,
            mBlurred, Nx,Ny, true/*drawAxis*/, mTitles[8].c_str() );
        SDL_RenderDrawMatrix( rpRenderer, mPlotPositions[9], 0,0,0,0,
            mMask, Nx,Ny, true/*drawAxis*/, mTitles[9].c_str() );

        /* highlight current step with a green frame */
        unsigned iPos = mCurrentFrame;
        if ( mCurrentFrame == 4 )
            iPos = 8;
        if ( mCurrentFrame == 5 )
            iPos = 9;
        if ( mCurrentFrame >= 6 )
            iPos = 4+(mCurrentFrame-6) % 4;
        SDL_Rect rect = mPlotPositions[iPos];
        rect.x -= 0.15*rect.w;
        rect.y -= 0.10*rect.h;
        rect.w *= 1.2;
        rect.h *= 1.25;
        SDL_SetRenderDrawColor( rpRenderer, 0,128,0,255 );
        SDL_RenderDrawThickRect( rpRenderer, rect, 3 );

        /* Plot errors */
        if ( mReconstructedErrors.size() > 0 )
        {
            const unsigned maxWidth = mPlotPositions[10].w;
            const unsigned size = std::min( maxWidth, (unsigned) mReconstructedErrors.size() );
            float * data = &mReconstructedErrors[0];
            if ( size == maxWidth )
                data += mReconstructedErrors.size()-maxWidth;

            SDL_SetRenderDrawColor( rpRenderer, 0,0,0,255 );
            SDL_RenderDrawHistogram( rpRenderer, mPlotPositions[10],
                0,0,0,0, data, size,
                0 /*binWidth*/, false /*fill*/, true /*drawAxis*/,
                "Log[ Sum_\\gamma |g'|**2 / N**2 ]" );
        }
    }

    void step( void )
    {
        using namespace imresh::math::image;
        using namespace imresh::math::vector;

        ++mCurrentFrame;

        /* define some aliases according to Fienup82 */
        const auto & F        = mImageState[1];
        const auto & absF     = mImageState[2];
        const auto & autocorr = mImageState[3]; // autocorrelation
        const auto & GPrime   = mImageState[4];
        const auto & gPrime   = mImageState[5];
        const auto & g        = mImageState[6];
        const auto & G        = mImageState[7];

        /* fourier transform the original image */
        if ( mCurrentFrame == 1 )
        {
            /* create and execute fftw plan */
            fftw_plan planForward = fftw_plan_dft_2d( Nx,Ny,
                mImageState[0] /*f*/, mImageState[1] /*F*/,
                FFTW_FORWARD, FFTW_ESTIMATE );
            fftw_execute(planForward);
            fftw_destroy_plan(planForward);
        }
        /* strip fourier transformed real image of it's phase (measurement) */
        else if ( mCurrentFrame == 2 )
        {
            for ( unsigned i = 0; i < Nx*Ny; ++i )
            {
                const float & re = F[i][0]; /* Re */
                const float & im = F[i][1]; /* Im */
                absF[i][0] = sqrtf( re*re + im*im ); /* Re */
                absF[i][1] = 0;  /* Im */
            }
        }
        else if ( mCurrentFrame == 3 )
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
            const float absMax = vectorMaxAbs( mBlurred, Nx*Ny );
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

};



int main(void)
{
    using namespace sdlcommon;

    SDL_Window   * pWindow;
    SDL_Renderer * pRenderer;

    /* Initialize SDL Context */
    SDL_CHECK( SDL_Init( SDL_INIT_VIDEO ) )

    pWindow = SDL_CreateWindow( "Understand Shrink-Wrap",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        1150, 650, SDL_WINDOW_SHOWN );
    SDL_CHECK( pWindow );

    pRenderer = SDL_CreateRenderer( pWindow, -1, SDL_RENDERER_ACCELERATED );
    SDL_CHECK( pRenderer );

    using namespace imresh::examples;

#if false
    const unsigned Nx = 40, Ny = 40;
    float * example = createVerticalSingleSlit( Nx, Ny );
    AnimateHybridInputOutput animateHybridInputOutput( example, Nx, Ny );
#else
    const unsigned Nx = 160, Ny = 160;
    float * example = createAtomCluster( Nx, Ny );
    AnimateHybridInputOutput animateHybridInputOutput( example, Nx, Ny );
#endif
    delete[] example;

    animateHybridInputOutput.step();
    animateHybridInputOutput.render( pRenderer );


    /* Wait for key to quit */
    int mainProgrammRunning = 1;
    int renderTouched = 1;
    while (mainProgrammRunning)
    {
        /* Handle Keyboard and Mouse events */
        SDL_Event event;
        while ( SDL_PollEvent( &event ) )
        {
            mainProgrammRunning &= not SDL_basicControl(event,pWindow,pRenderer);
            bool drawNext = SDL_animControl( event );
            if ( drawNext )
            {
                animateHybridInputOutput.step();
                renderTouched = 1;
            }
        }
        bool drawNext = SDL_animControl( event );
        if ( drawNext )
        {
            animateHybridInputOutput.step();
            renderTouched = 1;
        }

        if ( renderTouched )
        {
            renderTouched = 0;
            /* clear screen */
            SDL_SetRenderDrawColor( pRenderer, 255,255,255,255 );
            SDL_RenderClear( pRenderer );
            /* render and display */
            animateHybridInputOutput.render( pRenderer );
            SDL_RenderPresent( pRenderer );
        }
        SDL_Delay(50 /*ms*/);
    }

	SDL_DestroyWindow( pWindow );
	SDL_Quit();
    return 0;
}
