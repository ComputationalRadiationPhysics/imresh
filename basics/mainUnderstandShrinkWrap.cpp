
#include <iostream>
#include <complex>
#include <cstdlib>
#include <cmath>
#include <string>
#include <fftw3.h>
#include "sdl/sdlcommon.h"
#include "sdl/sdlplot.h"
#include "../examples/createAtomCluster.cpp"
#include "../examples/createSlit.cpp"
#include "math/vector/vectorReduce.h"

#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif


/**
 * Maps hue(H), saturation(S), value(V) to red, green, blue
 *
 * @verbatim
 *                                                  ^
 *                                               V _|         _______
 *                                       ^          |        /       \
 *                                     S |          |       /         \
 *                                       v  V(1-S) _|______/   Blue    \
 *                                ^                 |
 *                             V _|   ______        +------------------> Hue
 *                                |  /      \       0  1  2  3  4  5  6  * Pi/3
 *                                | /        \
 *          ^             V(1-S) _|/   Green  \______
 *       V _|___            ___   |
 *          |   \   Red    /      +------------------> Hue
 *          |    \        /       0  1  2  3  4  5  6  * Pi/3
 *  V(1-S) _|     \______/
 *          |
 *          +------------------> Hue
 *          0  1  2  3  4  5  6  * Pi/3
 * @endverbatim
 *
 * @param[in] hue assumed to be in [0,2*Pi]. 0: red, 2*Pi/3: green, 4*Pi/4: blue
 *            The hue cycles through all colors.
 * @param[in] saturation assumed to be in [0,1]. This value affects the range
 *            of possible colors (when varying the other 2 parameters).
 *            If 0, then only black will be returned.
 *            If 1, then the full color spectrum can be assumend.
 * @param[in] value assumed to be in [0,1]. For fixed hue and saturation this
 *            basically determines the how many white/light colors we are able
 *            to reach. It's like a top offset. For v=0 everything is black,
 *            but for v=1 we can get the full spectrum, including pure white.
 * @param[out] red will be in [0,1]
 * @param[out] green will be in [0,1]
 * @param[out] blue will be in [0,1]
 **/
void hsvToRgb
( const float hue, const float saturation, const float value,
  float * red, float * green, float * blue )
{
    /**
     * This is the trapeze function of the green channel. The other channel
     * functions can be derived by calling this function with a shift.
     **/
    struct { float operator()( float rHue, float rSat, float rVal )
    {
        /* rHue will be in [0,6]. Note that fmod(-5.1,3.0) = -2.1 */
        rHue = fmod( rHue, 6 );
        if ( rHue < 0 )
            rHue += 6.0;
        /*        _____              __             __        *
         *       /            -  ___/        =     /  \__     */
        float hue = fmin( 1,rHue ) - fmax( 0, fmin( 1,rHue-3 ) );
        return rVal*( (1-rSat) + rSat*hue );
    } } trapeze;

    *red   = trapeze( hue / (M_PI/3) + 2, saturation, value );
    *green = trapeze( hue / (M_PI/3)    , saturation, value );
    *blue  = trapeze( hue / (M_PI/3) + 4, saturation, value );
}

/**
 * Maps hue, saturation and lightes not red, green, blue
 *
 * The main advantage in contrast to HSV is, that the luminosity will
 * go from black/dark (0) to white/bright (1) instead of the value which
 * goes from black/dark (0) to maximally colorful (1) which would
 * correspond to luminosity = 0.5
 *
 * @param[in] hue assumed to be in [0,2*Pi]. 0: red, 2*Pi/3: green, 4*Pi/4: blue
 *            The hue cycles through all colors.
 * @param[in] saturation assumed to be in [0,1]. This value affects the range
 *            of possible colors (when varying the other 2 parameters).
 *            If 0, then only black will be returned.
 *            If 1, then the full color spectrum can be assumed.
 * @param[in] luminosity assumed to be in [0,1]
 * @param[out] red will be in [0,1]
 * @param[out] green will be in [0,1]
 * @param[out] blue will be in [0,1]
 **/
void hslToRgb
( const float hue, const float saturation, const float luminosity,
  float * red, float * green, float * blue )
{
    /**
     * This mapping from HSL to HSV coordinates is derived, seeing that the
     * formulae for HSV and HSL are very similar especially the hue:
     * @see https://en.wikipedia.org/w/index.php?title=HSL_and_HSV&oldid=687890438#Converting_to_RGB
     * Equating the intermediary values we get:
     *          H ... hue                  H ... hue
     *          S ... HSV-saturation       T ... HSL-saturation
     *          V ... value                L ... luminosity
     *   C = (1-|2L-1|) T = V S      (1)
     *   m = L - C/2      = V - C    (2)
     * Solving this system of equations for V(L,T) and S(L,T) we get:
     *   (1) => S(L,T) = C(L,T) T / V
     *   (2) => V(L,T) = C(L,T) T + L
     *
     *        chroma
     *          ^_____ saturation
     *          | /\
     *          |/  \
     *          +----> luminosity
     *          0    1
     *
     * Note that the HSL-formula maps to a hexcone instead of a circular cone,
     * like it also can be read in literature!
     * This should not be the standard behavior, but it is easier.
     **/
    const float chroma = ( 1-fabs(2*luminosity-1) )*saturation;
    const float value  = chroma/2 + luminosity;
    const float hsvSat = chroma/value;
    hsvToRgb( hue, hsvSat, value, red, green, blue );
}

void testHsv( SDL_Renderer * rpRenderer )
{
    using namespace sdlcommon;

    SDL_Rect rect = { 40,40, 100, 100 };
    for ( float saturation = 0.4; saturation <= 1.0; saturation += 0.2 )
    {
        SDL_SetRenderDrawColor( rpRenderer, 0,0,0,1 );
        int w = 3;
        SDL_Rect boundingBox = SDL_Rect{ rect.x-w, rect.y-w, rect.w+2*w, rect.h+2*w };
        SDL_RenderDrawThickRect( rpRenderer, &boundingBox, w );

        for ( int ix = 0; ix < rect.w; ++ix )
        for ( int iy = 0; iy < rect.h; ++iy )
        {
            float r,g,b;
            hsvToRgb( (float)ix/rect.w * 2*M_PI, saturation, (float)iy/rect.h,
                      &r,&g,&b );
            SDL_SetRenderDrawColor( rpRenderer, 255*r,255*g,255*b, /*a*/ 1 );
            //std::cout << "r="<<r<<", g="<<g<<", b="<<b<<"\n";
            SDL_RenderDrawPoint( rpRenderer, rect.x+ix, rect.y+(rect.h-1)-iy );
        }
        rect.x += 40 + rect.w;
    }
}

void testHsl( SDL_Renderer * rpRenderer )
{
    using namespace sdlcommon;

    SDL_Rect rect = { 40,180, 100, 100 };
    for ( float saturation = 0.4; saturation <= 1.0; saturation += 0.2 )
    {
        SDL_SetRenderDrawColor( rpRenderer, 0,0,0,1 );
        int w = 3;
        SDL_Rect boundingBox = SDL_Rect{ rect.x-w, rect.y-w, rect.w+2*w, rect.h+2*w };
        SDL_RenderDrawThickRect( rpRenderer, &boundingBox, w );

        for ( int ix = 0; ix < rect.w; ++ix )
        for ( int iy = 0; iy < rect.h; ++iy )
        {
            float r,g,b;
            hslToRgb( (float)ix/rect.w * 2*M_PI, saturation, (float)iy/rect.h,
                      &r,&g,&b );
            SDL_SetRenderDrawColor( rpRenderer, 255*r,255*g,255*b, /*a*/ 1 );
            //std::cout << "r="<<r<<", g="<<g<<", b="<<b<<"\n";
            SDL_RenderDrawPoint( rpRenderer, rect.x+ix, rect.y+rect.h-iy );
        }
        rect.x += 40 + rect.w;
    }
}

/**
 * Uses domain coloring to plot a complex valued matrix
 *
 * The matrix could e.g. contain evaluations of a complex function.
 *
 * @param[in] colorFunction 1:HSL (H=arg(z), S=1, L=|z|)
 *                          2:HSV
 *                          3:
 * @param[in] swapQuadrants true: rows and columns will be shifted by half
 *            width thereby centering the shortest wavelengths instead of
 *            those being at the corners
 **/
void SDL_RenderDrawComplexMatrix
(
  SDL_Renderer * const rpRenderer, const SDL_Rect & rAxes,
  const float x0, const float x1, const float y0, const float y1,
  fftw_complex * const values, const unsigned nValuesX, const unsigned nValuesY,
  const bool drawAxis, const char * const title,
  const bool swapQuadrants = false, const int colorFunction = 2 )
{
    using namespace sdlcommon;
    using namespace imresh::math::vector;

    const unsigned dataSize = /*rgb*/ 3*sizeof(float)*nValuesX*nValuesY;
    float * toPlot = (float*) malloc( dataSize );

    /* find maximum magnitude (which is always positive) to find out
     * how to scale to [0,1] */
    float maxMagnitude = 0;
    for ( unsigned i = 0; i < nValuesX*nValuesY; ++i )
    {
        const float & re = values[i][0];
        const float & im = values[i][1];
        maxMagnitude = fmax( maxMagnitude, sqrtf( re*re + im*im ) );
    }

    /* convert complex numbers to a color value to plot using */
    for ( unsigned ix = 0; ix < nValuesX; ++ix )
    for ( unsigned iy = 0; iy < nValuesY; ++iy )
    {
        /**
         * for the 1D case the fouriertransform looks like:
         *   @f[ \forall k = 0\ldots N: \tilde{x}_k = \sum\limits{n=0}^{N-1}
             x_n e^{  -2 \pi k \frac{n}{N} } @f]
         * This means for k=0, meaning the first element in the result error
         * will contain the sum over the function. the value k=1 will contain
         * the sum of f(x)*sin(x). Because the argument of exl(ix) is periodic
         * the last element in the array k=N-1 is equal to k=-1 which is the
         * sum over f(x)*sin(-x). This value will be similarily large as k=1.
         * This means the center of the array will contain the smallest
         * coefficients because those are high frequency coefficients.
         * The problem now is, that normally the diffraction pattern actually
         * goes from k=-infinity to infinity, meaning k=0 lies in the middle.
         * Because the discrete fourier transform is periodic the center is
         * arbitrary.
         * In order to reach a real diffraction pattern we need to shift k=0 to
         * the center of the array before plotting. In 2D this applies to both
         * axes:
         * @verbatim
         *        +------------+      +------------+      +------------+
         *        |            |      |## ++  ++ ##|      |     --     |
         *        |            |      |o> ''  '' <o|      | .. <oo> .. |
         *        |     #      |  FT  |-          -|      | ++ #### ++ |
         *        |     #      |  ->  |-          -|  ->  | ++ #### ++ |
         *        |            |      |o> ..  .. <o|      | '' <oo> '' |
         *        |            |      |## ++  ++ ##|      |     --     |
         *        +------------+      +------------+      +------------+
         *                           k=0         k=N-1         k=0
         * @endverbatim
         * This index shift can be done by a simple shift followed by a modulo:
         *   newArray[i] = array[ (i+N/2)%N ]
         **/
        int index;
        if ( swapQuadrants == true )
            index = ( ( iy+nValuesY/2 ) % nValuesY ) * nValuesX +
                    ( ( ix+nValuesX/2 ) % nValuesX );
        else
            index = iy*nValuesX + ix;
        const std::complex<double> z = {
            values[index][0],
            values[index][1]
        };

        float magnitude = std::abs(z) / maxMagnitude;
        float phase     = std::arg(z);
        if ( phase < 0 ) phase += 2*M_PI;
        bool logScale = true;
        if ( logScale )
            magnitude = log( 1+std::abs(z) ) / log( 1+maxMagnitude );

        /* convert magnitude and phase to color */
        float & r = toPlot[ ( iy*nValuesX + ix )*3 + 0 ];
        float & g = toPlot[ ( iy*nValuesX + ix )*3 + 1 ];
        float & b = toPlot[ ( iy*nValuesX + ix )*3 + 2 ];
        switch( colorFunction )
        {
            case 1:
                hslToRgb( phase, 1, magnitude, &r, &g, &b );
                break;
            case 2:
                hsvToRgb( 0, 1, magnitude, &r, &g, &b );
                break;
            case 3:
                /* we can't use black because else for phi = 0 everything
                 * would be black, no matter the magnitude!
                 * phi = 0      : 196 196 196
                 * phi = 2*pi/3 : 0   196 0    (darker green)   ^ basically
                 * phi = 4*pi/3 : 0   196 196  (turquese)       | hsv from
                 * phi = 6*pi/3 : 196 196 0    (darker yellow)  v [2*pi,5*pi]
                 */
                float saturation = 196.0f/255.0f;
                float interval = 2*M_PI/3;
                float pmod = fmod( phase, interval ) / interval;

                if ( phase < 2*M_PI/3 )
                {
                    r = saturation*(1-pmod);
                    g = saturation;
                    b = saturation*(1-pmod);
                }
                else if ( phase < 4*M_PI/3 )
                {
                    r = 0;
                    g = saturation;
                    b = saturation*pmod;
                }
                else if ( phase <= 2*M_PI+1e-3 )
                {
                    r = saturation * pmod;
                    g = saturation;
                    b = saturation*(1-pmod);
                }
                else
                    assert(false);

                r *= magnitude;
                g *= magnitude;
                b *= magnitude;

                break;
        }
    }

    SDL_RenderDrawMatrix( rpRenderer, rAxes, x0,y0,x1,y1,
        toPlot,nValuesX,nValuesY, drawAxis, title, true /* useColors */ );

    free( toPlot );
}

class AnimateShrinkWrap
{
private:
    int mCurrentFrame;

    const unsigned Nx, Ny;
    static constexpr unsigned mnSteps = 6;
    SDL_Rect mPlotPositions[ mnSteps ];
    fftw_complex * mImageState[ mnSteps ];
    std::string mTitles[mnSteps];

    typedef struct { int x0,y0,x1,y1; } Line2d;
    static constexpr unsigned mnArrows = 6;
    Line2d mArrows[ mnArrows ];
    /**
     *         +---+    +---+    +---+
     *         | 1 | -> | 2 | -> | 3 |
     *         +---+    +---+    +---+
     *           ^        ^        |
     *           |        |        v
     *         +---+    +---+    +---+
     *         | 0 |    | 5 | <- | 4 |
     *         +---+    +---+    +---+
     *
     * 0 ... original image (normally we don't have that, but we want to
     *       reconstruct this
     * 1 ... complex fourier transformed original image
     * 2 ... current guess for the measured intensity. In the first step this
     *       is exactly the measured intensity
     * 3 ... fourier transformed measured image. In the first step the measured
     *       intensity is real valued. In that case this is the autocorrellation
     *       map
     **/

public:
    AnimateShrinkWrap
    ( float * const rpOriginalData, const unsigned rNx, const unsigned rNy )
     : mCurrentFrame(-1), Nx(rNx), Ny(rNy)
    {
        for ( unsigned i = 0; i < mnSteps; ++i )
        {
            mImageState[i] = fftw_alloc_complex(Nx*Ny);
            memset( mImageState[i], 0, sizeof(mImageState[i][0])*Nx*Ny );
        }

        /* save original image to first array */
        for ( unsigned i = 0; i < Nx*Ny; ++i )
        {
            mImageState[0][i][0] = rpOriginalData[i]; /* Re */
            mImageState[0][i][1] = 0; /* Im */
        }

        /* scale very small images up to a height of at least 100 */
        const int multiplierX = (int) ceilf( 200.0f / (float) Nx );
        const int multiplierY = (int) ceilf( 200.0f / (float) Ny );
        const int plotWidth  = multiplierX * Nx;
        const int plotHeight = multiplierY * Ny;

        /* Initialize plot titles */
        mTitles[0] = "Original Image";
        mTitles[1] = "FT[Original Image]";
        mTitles[2] = "Diffraction Intensity";
        mTitles[3] = "";
        mTitles[4] = "";
        mTitles[5] = "";

        /* Initialize plot positions */
        SDL_Rect tmp = { 40, 40 + int(1.5*plotHeight), plotWidth, plotHeight };
        mPlotPositions[0] = tmp;
        tmp.y -= 1.5*plotHeight;    mPlotPositions[1] = tmp;
        tmp.x += 1.5*plotWidth;     mPlotPositions[2] = tmp;
        tmp.x += 1.5*plotWidth;     mPlotPositions[3] = tmp;
        tmp.y += 1.5*plotHeight;    mPlotPositions[4] = tmp;
        tmp.x -= 1.5*plotWidth;     mPlotPositions[5] = tmp;

        /* Initialize arrows */
        /* up */
        tmp = mPlotPositions[0];
        mArrows[0].x0 = tmp.x + tmp.w/2;
        mArrows[0].x1 = mArrows[0].x0;
        mArrows[0].y0 = tmp.y - 0.1*tmp.h;
        mArrows[0].y1 = mArrows[0].y0 - 0.3*tmp.h;
        /* -> */
        tmp = mPlotPositions[1];
        mArrows[1].x0 = tmp.x + 1.1*tmp.w;
        mArrows[1].x1 = mArrows[1].x0 + 0.3*tmp.w;
        mArrows[1].y0 = tmp.y + tmp.h/2;
        mArrows[1].y1 = mArrows[1].y0;
        /* -> */
        tmp = mPlotPositions[2];
        mArrows[2].x0 = tmp.x + 1.1*tmp.w;
        mArrows[2].x1 = mArrows[2].x0 + 0.3*tmp.w;
        mArrows[2].y0 = tmp.y + tmp.h/2;
        mArrows[2].y1 = mArrows[2].y0;
        /* down */
        tmp = mPlotPositions[3];
        mArrows[3].x0 = tmp.x + tmp.w/2;
        mArrows[3].x1 = mArrows[3].x0;
        mArrows[3].y0 = tmp.y + 1.1*tmp.h;
        mArrows[3].y1 = mArrows[3].y0 + 0.3*tmp.h;
        /* <- */
        tmp = mPlotPositions[4];
        mArrows[4].x0 = tmp.x - 0.1*tmp.w;
        mArrows[4].x1 = mArrows[4].x0 - 0.3*tmp.w;
        mArrows[4].y0 = tmp.y + tmp.h/2;
        mArrows[4].y1 = mArrows[4].y0;
        /* up */
        tmp = mPlotPositions[5];
        mArrows[5].x0 = tmp.x + tmp.w/2;
        mArrows[5].x1 = mArrows[5].x0;
        mArrows[5].y0 = tmp.y - 0.1*tmp.h;
        mArrows[5].y1 = mArrows[5].y0 - 0.3*tmp.h;
    }

    ~AnimateShrinkWrap()
    {
        for ( unsigned i = 0; i < mnSteps; ++i )
            fftw_free( mImageState[i] );
    }

    void render( SDL_Renderer * rpRenderer )
    {
        using namespace sdlcommon;

        /* Draw arrows and complex plots */
        for ( unsigned i = 0; i < mnArrows; ++i )
        {
            const Line2d & l = mArrows[i];
            SDL_RenderDrawArrow( rpRenderer, l.x0, l.y0, l.x1, l.y1 );
        }
        for ( unsigned i = 0; i < 4/*mnSteps*/; ++i )
        {
            SDL_RenderDrawComplexMatrix( rpRenderer, mPlotPositions[i], 0,0,0,0,
                mImageState[i],Nx,Ny, true/*drawAxis*/, mTitles[i].c_str(),
                i != 0 and i != 3 /* swapQuadrants */, 1 /* color map */ );
        }
    }

    void step( void )
    {
        ++mCurrentFrame;

        /* we already have the original image, nothing to be done here */
        if ( mCurrentFrame == 0 )
            return;
        /* fourier transform the original image */
        else if ( mCurrentFrame == 1 )
        {
            /* create and execute fftw plan */
            fftw_plan planForward = fftw_plan_dft_2d( Nx,Ny,
                mImageState[0], mImageState[1], FFTW_FORWARD, FFTW_ESTIMATE );
            fftw_execute(planForward);
            fftw_destroy_plan(planForward);
        }
        /* strip fourier transformed real image of it's phase (measurement) */
        else if ( mCurrentFrame == 2 )
        {
            for ( unsigned i = 0; i < Nx*Ny; ++i )
            {
                const float & re = mImageState[1][i][0]; /* Re */
                const float & im = mImageState[1][i][1]; /* Im */
                mImageState[2][i][0] = sqrtf( re*re + im*im ); /* Re */
                mImageState[2][i][1] = 0;  /* Im */
            }
        }
        else if ( (mCurrentFrame-3) % 4 == 0 )
        {
            std::cout << "do inverse FT\n";
            /* create and execute fftw plan */
            fftw_plan fft = fftw_plan_dft_2d( Nx,Ny,
                mImageState[2], mImageState[3], FFTW_BACKWARD, FFTW_ESTIMATE );
            fftw_execute(fft);
            fftw_destroy_plan(fft);
        }

    };

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
        1024, 960, SDL_WINDOW_SHOWN );
    SDL_CHECK( pWindow );

    pRenderer = SDL_CreateRenderer( pWindow, -1, SDL_RENDERER_ACCELERATED );
    SDL_CHECK( pRenderer );

    SDL_SetRenderDrawColor( pRenderer, 255,255,255,255 );
    SDL_RenderClear( pRenderer );
    SDL_RenderPresent( pRenderer );

    /*
    testHsv( pRenderer );
    testHsl( pRenderer );
    */

    using namespace imresh::examples;

#if false
    const unsigned Nx = 50, Ny = 50;
    float * example = createVerticalSingleSlit( Nx, Ny );
    AnimateShrinkWrap animateShrinkWrap( example, Nx, Ny );
#else
    const unsigned Nx = 200, Ny = 300;
    float * example = createAtomCluster( Nx, Ny );
    AnimateShrinkWrap animateShrinkWrap( example, Nx, Ny );
#endif
    delete[] example;

    animateShrinkWrap.step();
    animateShrinkWrap.step();
    animateShrinkWrap.step();
    animateShrinkWrap.step();
    SDL_SetRenderDrawColor( pRenderer, 0,0,0,255 );
    animateShrinkWrap.render(pRenderer);


    /* Wait for key to quit */
    int mainProgrammRunning = 1;
    int renderTouched = 1;
    while (mainProgrammRunning)
    {
        /* Handle Keyboard and Mouse events */
        SDL_Event event;
        while ( SDL_PollEvent(&event) )
        {
            mainProgrammRunning &= not SDL_basicControl(event,pWindow,pRenderer);
            SDL_SetRenderDrawColor( pRenderer, 128,0,0,255 );
            //renderTouched |= drawControl(event, pRenderer);
        }

        if ( renderTouched )
        {
            renderTouched = 0;
            SDL_RenderPresent( pRenderer );
        }
        SDL_Delay(50 /*ms*/);
    }

	SDL_DestroyWindow( pWindow );
	SDL_Quit();
    return 0;
}
