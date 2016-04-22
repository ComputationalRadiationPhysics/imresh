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

#include "algorithms/vectorReduce.hpp" // vectorMax


namespace imresh
{
namespace io
{
namespace writeOutFuncs
{


    void justFree
    (
        float * _mem,
        unsigned int const imageWidth,
        unsigned int const imageHeight,
        std::string const _filename
    )
    {
        if( _mem != NULL )
        {
            delete[] _mem;
        }
#       ifdef IMRESH_DEBUG
            std::cout << "imresh::io::writeOutFuncs::justFree(): Freeing data ("
                << _filename << ")." << std::endl;
#       endif
    }

#   ifdef USE_PNG
        void writeOutPNG
        (
            float * _mem,
            unsigned int const imageWidth,
            unsigned int const imageHeight,
            std::string const _filename
        )
        {
            auto _size = std::make_pair( imageWidth, imageHeight );
            pngwriter png( _size.first, _size.second, 0, _filename.c_str( ) );

            float max = algorithms::vectorMax( _mem,
                                        _size.first * _size.second );
            /* avoid NaN in border case where all values are 0 */
            if ( max == 0 )
                max = 1;
            for( unsigned int iy = 0; iy < _size.second; ++iy )
            {
                for( unsigned int ix = 0; ix < _size.first; ++ix )
                {
                    auto const index = iy * _size.first + ix;
                    assert( index < _size.first * _size.second );
                    auto const value = _mem[index] / max;
#                   ifdef IMRESH_DEBUG
                    if ( isnan(value) or isinf(value) or value < 0 )
                    {
                        /* write out red  pixel to alert user that something is wrong */
                        png.plot( 1+ix, 1+iy,
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

#           ifdef IMRESH_DEBUG
                std::cout << "imresh::io::writeOutFuncs::writeOutPNG(): "
                             "Successfully written image data to PNG ("
                          << _filename << ")." << std::endl;
#           endif
        }

        void writeOutAndFreePNG
        (
            float * _mem,
            unsigned int const imageWidth,
            unsigned int const imageHeight,
            std::string const _filename
        )
        {
            writeOutPNG( _mem, imageWidth, imageHeight, _filename );
            justFree   ( _mem, imageWidth, imageHeight, _filename );
        }
#   endif

#   ifdef USE_SPLASH
        void writeOutHDF5
        (
            float * _mem,
            unsigned int const imageWidth,
            unsigned int const imageHeight,
            std::string const _filename
        )
        {
            splash::SerialDataCollector sdc( 0 );
            splash::DataCollector::FileCreationAttr fCAttr;
            splash::DataCollector::initFileCreationAttr( fCAttr );

            fCAttr.fileAccType = splash::DataCollector::FAT_CREATE;

            sdc.open( _filename.c_str( ), fCAttr );

            splash::ColTypeFloat cTFloat;
            splash::Dimensions size( imageWidth, imageHeight, 1 );

            sdc.write( 0,
                       cTFloat,
                       2,
                       splash::Selection( size ),
                       _filename.c_str( ),
                       _mem );

            sdc.close( );

#           ifdef IMRESH_DEBUG
                std::cout << "imresh::io::writeOutFuncs::writeOutHDF5(): "
                             "Successfully written image data to HDF5 ("
                          << _filename << "_0_0_0.h5)." << std::endl;
#           endif
        }

        void writeOutAndFreeHDF5
        (
            float * _mem,
            unsigned int const imageWidth,
            unsigned int const imageHeight,
            std::string const _filename
        )
        {
            writeOutHDF5( _mem, imageWidth, imageHeight, _filename );
            justFree    ( _mem, imageWidth, imageHeight, _filename );
        }
#   endif


} // namespace writeOutFuncs
} // namespace io
} // namespace imresh

