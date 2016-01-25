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


#include "createCircularSection.hpp"

#include <cmath>    // atan2
#include <cassert>

#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif


namespace examples
{
namespace createTestData
{


    float * createCircularSection
    (
        unsigned const & Nx,
        unsigned const & Ny,
        float    const & r,
        float    const & x0,
        float    const & y0,
        float    const & phi0,
        float    const & phi1
    )
    {
        assert( 0.0f <= r );
        assert( 0.0f <= x0 and x0 <= 1.0f );
        assert( 0.0f <= y0 and y0 <= 1.0f );
        assert( phi0 <= phi1 );

        float * data = new float[ Nx*Ny ];
        const unsigned Nmin = ( Nx < Ny ) ? Nx : Ny;

        for ( unsigned iy = 0; iy < Ny; ++iy )
        for ( unsigned ix = 0; ix < Nx; ++ix )
        {
            float x = (float) ix / Nmin - x0;
            float y = (float) iy / Nmin - y0;
            float r2  = x*x + y*y;
            float phi = M_PI + atan2( y, x );

            if ( phi0 <= phi and phi <= phi1 and r2 <= r*r )
                data[iy*Nx + ix] = 1;
            else
                data[iy*Nx + ix] = 0;
        }

        return data;
    }


} // namespace createTestData
} // namespace examples
