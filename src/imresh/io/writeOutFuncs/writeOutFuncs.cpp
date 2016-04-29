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

#include "algorithms/vectorReduce.hpp" // vectorMax
#include "io/writeOutFuncs/writeOutFuncs.hpp"


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


} // namespace writeOutFuncs
} // namespace io
} // namespace imresh

