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


#include "testHybridInputOutput.h"


namespace imresh
{
namespace test
{


    void testHybridInputOutput
    ( SDL_Renderer * const & rpRenderer )
    {
        using namespace sdlcommon;
        using namespace imresh::examples;
        using namespace imresh::algorithms::phasereconstruction;

        std::vector<unsigned> imageSize = {40,40};
        const unsigned & Nx = imageSize[1];
        const unsigned & Ny = imageSize[0];

        /* display rectangle */
        float * rectangle = createVerticalSingleSlit( Nx, Ny );
        SDL_Rect position = { 40,30,160,160 };
        SDL_RenderDrawMatrix( rpRenderer, position,0,0,0,0, rectangle,Nx,Ny,
            true /*drawAxis*/, "Rectangular Opening" );
        SDL_RenderDrawArrow( rpRenderer,
            position.x + 1.1*position.w, position.y + position.h/2,
            position.x + 1.4*position.w, position.y + position.h/2 );
        position.x += 1.5*position.w;

        /* create mask for HIO */
        uint8_t * mask = new uint8_t[ Nx*Ny ];
        for ( unsigned i = 0; i < Nx*Ny; ++i )
            mask[i] = rectangle[i];

        /* convert to diffraction pattern and display it */
        algorithms::diffractionIntensity( rectangle, imageSize );



        //for ( unsigned i = 0; i < Nx*Ny; ++i )
        //    rectangle[i] = logf( 1+rectangle[i] );
        float max = 0;
        for ( unsigned i = 0; i < Nx*Ny; ++i )
            max = std::max( max, rectangle[i] );
        for ( unsigned i = 0; i < Nx*Ny; ++i )
            rectangle[i] /= max;
        SDL_RenderDrawMatrix( rpRenderer, position,0,0,0,0, rectangle,Nx,Ny,
            true /*drawAxis*/, "Diffraction Intensity" );

        /* -> */
        SDL_RenderDrawArrow( rpRenderer,
            position.x + 1.1*position.w, position.y + position.h/2,
            position.x + 1.4*position.w, position.y + position.h/2 );
        position.x += 1.5*position.w;

        /* display reconstructed image */
        int hioError = hybridInputOutput( rectangle, mask, imageSize,
            64 /*cycles*/, 1e-6 /* targetError */, 0.9 /* beta */, 0 /* cores */ );
        assert( hioError == 0 );
        {
            float max = 0;
            for ( unsigned i = 0; i < Nx*Ny; ++i )
                max = std::max( max, rectangle[i] );
            for ( unsigned i = 0; i < Nx*Ny; ++i )
                rectangle[i] /= max;
            SDL_RenderDrawMatrix( rpRenderer, position,0,0,0,0, rectangle,Nx,Ny,
                true /*drawAxis*/, "Reconstructed Object" );
        }


        delete[] rectangle;
        delete[] mask;
    }


} // namespace imresh
} // namespace test

