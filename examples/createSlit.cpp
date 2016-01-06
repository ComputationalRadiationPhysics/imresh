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


#include "createSlit.hpp"


namespace examples
{


float * createVerticalSingleSlit
(
    const unsigned & Nx,
    const unsigned & Ny
)
{
    float * data = new float[Nx*Ny];
    memset( data, 0, Nx*Ny*sizeof(float) );

    const int slitHalfHeight = (int) ceilf( 0.3*Nx );
    const int slitHalfWidth  = (int) ceilf( 0.1*Nx );
    for ( unsigned iy = Ny/2 - slitHalfHeight+1; iy < Ny/2 + slitHalfHeight; ++iy )
    for ( unsigned ix = Nx/2 - slitHalfWidth +1; ix < Nx/2 + slitHalfWidth ; ++ix )
        data[iy*Nx + ix] = 1;

    return data;
}


} // namespace examples
