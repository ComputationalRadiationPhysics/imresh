/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Philipp Trommler
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

#ifdef USE_PNG
#   include <pngwriter.h>
#endif
#ifdef USE_SPLASH
#   include <splash/splash.h>
#endif
#include <string>                   // std::string
#include <utility>                  // std::pair

#include "io/writeOutFuncs/writeOutFuncs.hpp"

namespace imresh
{
namespace io
{
namespace writeOutFuncs
{
    void justFree(
        float* _mem,
        std::pair<unsigned int,unsigned int> _size,
        std::string _filename
    )
    {
        delete _mem;
        _mem = NULL;
    }

#   ifdef USE_PNG
        void writeOutPNG(
            float* _mem,
            std::pair<unsigned int,unsigned int> _size,
            std::string _filename
        )
        {
            pngwriter png( _size.first, _size.second, 0, _filename.c_str( ) );

            for( auto i = 0; i < _size.first; i++ )
            {
                for( auto j = 0; j < _size.second; j++ )
                {
                    png.plot( i, j, _mem[(i * _size.second) + j],
                        _mem[(i * _size.second) + j],
                        _mem[(i * _size.second) + j] );
                }
            }

            png.close( );
        }
#   endif

#   ifdef USE_SPLASH
        void writeOutHDF5(
            float* _mem,
            std::pair<unsigned int,unsigned int> _size,
            std::string _filename
        )
        {
            // TODO
        }
#   endif
} // namespace writeOutFuncs
} // namespace io
} // namespace imresh

