/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Maximilian Knespel
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


#include "createCheckerboard.hpp"

#include <cmath>   // ceil
#include "rotateCoordinates.hpp"


namespace examples
{
namespace createTestData
{


    float * createCheckerboard
    (
        unsigned const & Nx,
        unsigned const & Ny,
        float    const & Dx,
        float    const & Dy,
        float    const & phi
    )
    {
        float * data = new float[ Nx*Ny ];

        for ( unsigned iy = 0; iy < Ny; ++iy )
        for ( unsigned ix = 0; ix < Nx; ++ix )
        {
            float x = ix, y = iy;
            rotateCoordinates2d( x,y, 0.5*Nx, 0.5*Ny, phi );
            if ( ( int( x / ( Dx * Nx ) + std::ceil( Nx / Dx ) ) ) % 2 == 0 and
                 ( int( y / ( Dy * Ny ) + std::ceil( Ny / Dy ) ) ) % 2 == 0 )
            {
                data[iy*Nx + ix] = 1;
            }
            else
            {
                data[iy*Nx + ix] = 0;
            }
        }

        return data;
    }


} // namespace createTestData
} // namespace examples
