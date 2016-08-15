/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Philipp Trommler, Maximilian Knespel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


#include "writeOutFuncs.hpp"

#include <algorithm>
#ifdef IMRESH_DEBUG
#   include <iostream>              // std::cout, std::endl

#endif
#ifdef USE_PNG
#   include <pngwriter.h>
#endif
#ifdef USE_SPLASH
#   include <splash/splash.h>
#endif
#include <string>                   // std::string
#include <utility>                  // std::pair
#include <cstddef>                  // NULL
#include <sstream>
#include <cassert>
#include <cmath>                    // isnan, isinf
#include <complex>

#include "algorithms/vectorReduce.hpp" // vectorMax


namespace imresh
{
namespace io
{
namespace writeOutFuncs
{


    void justFree
    (
        float * rMem,
        unsigned int const rImageWidth,
        unsigned int const rImageHeight,
        std::string const rFilename
    )
    {
        if( rMem != NULL )
            delete[] rMem;
    }

#   ifdef USE_PNG
        void writeOutPNG
        (
            float * rMem,
            unsigned int const rImageWidth,
            unsigned int const rImageHeight,
            std::string const rFileName
        )
        {
            pngwriter png( rImageWidth, rImageHeight, 0, rFileName.c_str( ) );

            float max = algorithms::vectorMax( rMem,
                                        rImageWidth * rImageHeight );
            /* avoid NaN in border case where all values are 0 */
            if ( max == 0 )
                max = 1;
            for( unsigned int iy = 0; iy < rImageHeight; ++iy )
            {
                for( unsigned int ix = 0; ix < rImageWidth; ++ix )
                {
                    auto const index = iy * rImageWidth + ix;
                    assert( index < rImageWidth * rImageHeight );
                    auto const value = rMem[index] / max;
#                   if IMRESH_DEBUG
                    if ( isnan(value) or isinf(value) or value < 0 )
                    {
                        /* write out red  pixel to alert user that something is wrong */
                        png.plot( 1 + ix, 1 + iy,
                                  65535 * isnan( value ), /* red */
                                  65535 * isinf( value ), /* green */
                                  65536 * ( value < 0 )   /* blue */ );
                    }
                    else
#                   endif
                    {
                        /* calling the double overloaded version with float
                         * values is problematic and will fail the unit test
                         * @see https://github.com/pngwriter/pngwriter/issues/83
                         *    v correct rounding for positive values */
                        int intToPlot = int( value*65535 + 0.5f );
                        png.plot( 1+ix, 1+iy, intToPlot, intToPlot, intToPlot );
                    }
                }
            }

            png.close( );
        }

        void writeOutAndFreePNG
        (
            float * rMem,
            unsigned int const rImageWidth,
            unsigned int const rImageHeight,
            std::string const rFileName
        )
        {
            writeOutPNG( rMem, rImageWidth, rImageHeight, rFileName );
            justFree   ( rMem, rImageWidth, rImageHeight, rFileName );
        }
#   endif

#   ifdef USE_SPLASH
        void writeOutHDF5
        (
            float * rMem,
            unsigned int const rImageWidth,
            unsigned int const rImageHeight,
            std::string const rFileName
        )
        {
            splash::SerialDataCollector sdc( 0 );
            splash::DataCollector::FileCreationAttr fCAttr;
            splash::DataCollector::initFileCreationAttr( fCAttr );

            fCAttr.fileAccType = splash::DataCollector::FAT_CREATE;

            sdc.open( rFileName.c_str( ), fCAttr );

            splash::ColTypeFloat cTFloat;
            splash::Dimensions size( rImageWidth, rImageHeight, 1 );

            sdc.write( 0,
                       cTFloat,
                       2,
                       splash::Selection( size ),
                       rFileName.c_str( ),
                       rMem );

            sdc.close( );
        }

        void writeOutAndFreeHDF5
        (
            float * rMem,
            unsigned int const rImageWidth,
            unsigned int const rImageHeight,
            std::string const rFileName
        )
        {
            writeOutHDF5( rMem, rImageWidth, rImageHeight, rFileName );
            justFree    ( rMem, rImageWidth, rImageHeight, rFileName );
        }
#   endif


#   ifdef USE_PNG
        template< typename T_Prec>
        void plotPng
        (
            T_Prec *     const rMem,
            unsigned int const rImageWidth,
            unsigned int const rImageHeight,
            std::string  const rFileName
        )
        {
            pngwriter png( rImageWidth, rImageHeight, 0, rFileName.c_str( ) );

            float max = algorithms::vectorMax( rMem,
                                        rImageWidth * rImageHeight );
            /* avoid NaN in border case where all values are 0 */
            if ( max == 0 )
                max = 1;
            for ( auto iy = 0u; iy < rImageHeight; ++iy )
            for ( auto ix = 0u; ix < rImageWidth ; ++ix )
            {
                auto const index = iy * rImageWidth + ix;
                assert( index < rImageWidth * rImageHeight );
                auto const value = rMem[index] / max;
                int intToPlot = int( value*65535 + 0.5f );
                png.plot( 1+ix, 1+iy, intToPlot, intToPlot, intToPlot );
            }
            png.close( );
        }

        #define __INSTANTIATE( T_Prec )         \
        template void plotPng<T_Prec>           \
        (                                       \
            T_Prec *     const rMem,            \
            unsigned int const rImageWidth,     \
            unsigned int const rImageHeight,    \
            std::string  const rFileName        \
        );
        __INSTANTIATE( float  )
        __INSTANTIATE( double )
        #undef __INSTANTIATE
#   endif

    void hsvToRgb
    (
        float   const hue       ,
        float   const saturation,
        float   const value     ,
        float * const red       ,
        float * const green     ,
        float * const blue
    )
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
                 }
        } trapeze;

        *red   = trapeze( hue / (M_PI/3) + 2, saturation, value );
        *green = trapeze( hue / (M_PI/3), saturation, value );
        *blue  = trapeze( hue / (M_PI/3) + 4, saturation, value );
    }

    void hslToRgb
    (
        float   const hue       ,
        float   const saturation,
        float   const luminosity,
        float * const red       ,
        float * const green     ,
        float * const blue
    )
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
         *_____ saturation
         *          | /\
         *          |/  \
         *          +----> luminosity
         *          0    1
         *
         * Note that the HSL-formula maps to a hexcone instead of a circular cone,
         * like it also can be read in literature!
         * This should not be the standard behavior, but it is easier.
         **/
        float const chroma = ( 1-fabs(2*luminosity-1) )*saturation;
        float const value  = chroma/2 + luminosity;
        /* this ternary check is especially import for value = 0 where hsvSat=NaN */
        float const hsvSat = chroma/value <= 1.0f ? chroma/value : 0;
        hsvToRgb( hue, hsvSat, value, red, green, blue );
    }

#if defined( USE_PNG ) && defined( USE_FFTW )
    template< typename T_Complex>
    void plotComplexPng
    (
        T_Complex *     const values       ,
        unsigned int    const nValuesX     ,
        unsigned int    const nValuesY     ,
        std::string     const rFileName    ,
        bool            const logPlot      ,
        bool            const swapQuadrants,
        unsigned int    const upsize
    )
    {
        pngwriter png( nValuesX * upsize, nValuesY * upsize, 0, rFileName.c_str( ) );
        /* find maximum magnitude (which is always positive) to find out
         * how to scale to [0,1] */
        float maxMagnitude = 0;
        for ( unsigned i = 0; i < nValuesX*nValuesY; ++i )
        {
            float const & re = values[i][0];
            float const & im = values[i][1];
            maxMagnitude = std::max( maxMagnitude, std::sqrt( re*re + im*im ) );
        }

        /* convert complex numbers to a color value to plot using */
        for ( auto ix = 0u; ix < nValuesX; ++ix )
        for ( auto iy = 0u; iy < nValuesY; ++iy )
        {
            /**
             * for the 1D case the fouriertransform looks like:
             *   @f[ \forall k = 0\ldots N: \tilde{x}_k = \sum\limits{n=0}{N-1}
             * x_n e{  -2 \pi k \frac{n}{N} } @f]
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
            {
                index = ( ( iy+nValuesY/2 ) % nValuesY ) * nValuesX +
                        ( ( ix+nValuesX/2 ) % nValuesX );
            }
            else
                index = iy*nValuesX + ix;
            std::complex<double> const z = { values[index][0], values[index][1] };

            float magnitude = std::abs(z) / maxMagnitude;
            float phase     = std::arg(z);
            if ( phase < 0 ) phase += 2*M_PI;
            if ( logPlot )
                magnitude = log( 1+std::abs(z) ) / log( 1+maxMagnitude );

            /* convert magnitude and phase to color */
            float r,g,b;
            hslToRgb( phase, 1, magnitude, &r, &g, &b );
            for ( auto ilx = 0u; ilx < upsize; ++ilx )
            for ( auto ily = 0u; ily < upsize; ++ily )
            {
                png.plot( 1 + ix*upsize + ilx, 1 + iy*upsize + ily,
                          int( r * 65535 + 0.5f ),
                          int( g * 65535 + 0.5f ),
                          int( b * 65535 + 0.5f ) );
            }
        } // ix,iy for-loop
        png.close( );
    }

    #define __INSTANTIATE( T_Complex )          \
    template void plotComplexPng<T_Complex>     \
    (                                           \
        T_Complex *     const values       ,    \
        unsigned int    const nValuesX     ,    \
        unsigned int    const nValuesY     ,    \
        std::string     const rFileName    ,    \
        bool            const logPlot      ,    \
        bool            const swapQuadrants,    \
        unsigned int    const upsize            \
    );

    __INSTANTIATE( fftwf_complex )
    //__INSTANTIATE( fftw_complex )

    #undef __INSTANTIATE
#endif


} // namespace writeOutFuncs
} // namespace io
} // namespace imresh

