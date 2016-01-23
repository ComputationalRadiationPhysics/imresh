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


#include "createRectangle.hpp"

#include <cassert>
#include "rotateCoordinates.hpp"


namespace examples
{
namespace createTestData
{


    float * createRectangle
    (
        unsigned const & Nx,
        unsigned const & Ny,
        float    const & Dx,
        float    const & Dy,
        float    const & x0,
        float    const & y0,
        float    const & phi
    )
    {
        assert( 0.0f <= Dx and Dx <= 1.0f );
        assert( 0.0f <= Dy and Dy <= 1.0f );
        assert( 0.0f <= x0 and x0 <= 1.0f );
        assert( 0.0f <= y0 and y0 <= 1.0f );

        float * data = new float[ Nx*Ny ];

        const int yLow  = ( y0 - Dy/2 ) * Ny;
        const int yHigh = ( y0 + Dy/2 ) * Ny;
        const int xLow  = ( x0 - Dx/2 ) * Nx;
        const int xHigh = ( x0 + Dx/2 ) * Nx;

        for ( unsigned iy = 0; iy < Ny; ++iy )
        for ( unsigned ix = 0; ix < Nx; ++ix )
        {
            float x = ix, y = iy;
            rotateCoordinates2d( x,y, x0*Nx, y0*Ny, phi );
            if ( x >= xLow and x <= xHigh and y >= yLow and y <= yHigh )
                data[iy*Nx + ix] = 1;
            else
                data[iy*Nx + ix] = 0;
        }

        return data;
    }


} // namespace createTestData
} // namespace examples
