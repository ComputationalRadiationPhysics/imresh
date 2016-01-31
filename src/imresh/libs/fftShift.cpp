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


#include "fftShift.hpp"

#include <cstddef>      // NULL
#include <algorithm>    // swap
#include <cassert>
#ifdef USE_FFTW
#   include <fftw3.h>
#endif


namespace imresh
{
namespace libs
{


    template<class T_COMPLEX>
    void fftShift
    (
        T_COMPLEX * const rpData,
        unsigned int const Nx,
        unsigned int const Ny
    )
    {
        assert( rpData != NULL );
        /* only up to Ny/2 necessary, because wie use std::swap meaning we correct
         * two elements with 1 operation */
        for ( auto iy = 0u; iy < Ny/2; ++iy )
        for ( auto ix = 0u; ix < Nx; ++ix )
        {
            auto const index =
                ( ( iy+Ny/2 ) % Ny ) * Nx +
                ( ( ix+Nx/2 ) % Nx );
            std::swap( rpData[iy*Nx + ix], rpData[index] );
        }
    }

    /* explicit template instantiations */

    #define INSTANTIATE_fftShift( T_COMPLEX ) \
    template                                  \
    void fftShift<T_COMPLEX>                  \
    (                                         \
        T_COMPLEX * const rpData,             \
        unsigned int const rNx,               \
        unsigned int const rNy                \
    );

    #ifdef USE_FFTW
        INSTANTIATE_fftShift( fftwf_complex )
    #endif
    INSTANTIATE_fftShift( float )


} // namespace algorithms
} // namespace imresh
