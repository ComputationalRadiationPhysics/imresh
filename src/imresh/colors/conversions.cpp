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


#include "conversions.h"


namespace imresh
{
namespace colors
{


void hsvToRgb
( const float hue, const float saturation, const float value,
  float * const red, float * const green, float * const blue )
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

void hslToRgb
( const float hue, const float saturation, const float luminosity,
  float * const red, float * const green, float * const blue )
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
    /* this ternary check is especially import for value = 0 where hsvSat=NaN */
    const float hsvSat = chroma/value <= 1.0f ? chroma/value : 0;
    hsvToRgb( hue, hsvSat, value, red, green, blue );
}


} // namespace colors
} // namespace imresh
