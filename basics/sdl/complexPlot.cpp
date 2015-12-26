
#include "complexPlot.h"


namespace sdlcommon {


void SDL_RenderDrawComplexMatrix
(
  SDL_Renderer * const rpRenderer, const SDL_Rect & rAxes,
  const float x0, const float x1, const float y0, const float y1,
  fftw_complex * const values, const unsigned nValuesX, const unsigned nValuesY,
  const bool drawAxis, const char * const title,
  const bool logPlot, const bool swapQuadrants, const int colorFunction
)
{
    using namespace sdlcommon;
    //using namespace imresh::math::vector;

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
        if ( logPlot )
            magnitude = log( 1+std::abs(z) ) / log( 1+maxMagnitude );

        /* convert magnitude and phase to color */
        using namespace imresh::colors;
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


} // namespace sdlcommon
