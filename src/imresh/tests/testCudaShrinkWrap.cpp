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


#include "testCudaShrinkWrap.h"
#include <fftw3.h>   // fftw_complex


namespace imresh
{
namespace test
{


    void testCudaShrinkWrap
    ( SDL_Renderer * const & rpRenderer )
    {
        using namespace sdlcommon;
        using namespace imresh::examples;
        using namespace imresh::algorithms::cuda;

        std::vector<unsigned> imageSize = {160,160};
        const unsigned & Nx = imageSize[1];
        const unsigned & Ny = imageSize[0];

        /* display rectangle */
        //float * rectangle = createVerticalSingleSlit( Nx, Ny );
        float * rectangle = createAtomCluster( imageSize );
        SDL_Rect position = { 40,30,160,160 };
        if ( rpRenderer != NULL )
        {
            SDL_RenderDrawMatrix( rpRenderer, position,0,0,0,0, rectangle,Nx,Ny,
                true /*drawAxis*/, "Rectangular Opening" );
            SDL_RenderDrawArrow( rpRenderer,
                position.x + 1.1*position.w, position.y + position.h/2,
                position.x + 1.4*position.w, position.y + position.h/2 );
            position.x += 1.5*position.w;
        }
        float * originalRectangle = new float[Nx*Ny];
        memcpy( originalRectangle, rectangle, sizeof(float)*Nx*Ny );

        /* convert to diffraction pattern and display it */
        algorithms::diffractionIntensity( rectangle, imageSize );

        if ( rpRenderer != NULL )
        {
            fftw_complex * tmp = (fftw_complex*) malloc( sizeof(fftw_complex)*Nx*Ny );

            for ( unsigned i = 0; i < Nx*Ny; ++i )
            {
                tmp[i][0] = rectangle[i];
                tmp[i][1] = 0;
            }
            SDL_RenderDrawComplexMatrix( rpRenderer, position, 0,0,0,0,
                tmp,Nx,Ny, true /*drawAxis*/, "Diffraction Intensity",
                true /*logPlot*/, true /*swapQuadrants*/, 1 );
            /* -> */
            SDL_RenderDrawArrow( rpRenderer,
                position.x + 1.1*position.w, position.y + position.h/2,
                position.x + 1.4*position.w, position.y + position.h/2 );
            position.x += 1.5*position.w;

            free( tmp );
        }

        /* display reconstructed image */
        int shrinkWrapError = cudaShrinkWrap( rectangle, imageSize,
            64 /*cycles*/, 1e-6 /* targetError */ );
        assert( shrinkWrapError == 0 );
#if false
        /* check if result is correct */
        float avgScale = 0;
        unsigned elementsNeqZero = 0;
        for ( unsigned i = 0; i < Nx*Ny; ++i )
        {
            if ( originalRectangle[i] != 0 )
            {
                avgScale += rectangle[i] / originalRectangle[i];
                ++elementsNeqZero;
            }
        }
        avgScale /= elementsNeqZero;
        //std::cout << "scaling factor is " << avgScale << " != " << Nx*Ny << "\n";
        float avgError = 0;
        elementsNeqZero = 0;
        for ( unsigned i = 0; i < Nx*Ny; ++i )
        {
            if ( originalRectangle[i] != 0 )
            {
                //std::cout << originalRectangle[i] << " - " << rectangle[i] << " = ";
                avgError += fabs( originalRectangle[i] - rectangle[i] / avgScale );
                //std::cout << "avgError = " << fabs( originalRectangle[i] / avgScale - rectangle[i] ) << "\n";
                ++elementsNeqZero;
            }
        }
        avgError /= elementsNeqZero;
        std::cout << "avgError = " << avgError << "\n";
        assert( avgError < 1e-3 );
#endif

        /* plot reconstructed image */
        if ( rpRenderer != NULL )
        {
            float max = 0;
            for ( unsigned i = 0; i < Nx*Ny; ++i )
                max = std::max( max, rectangle[i] );
            for ( unsigned i = 0; i < Nx*Ny; ++i )
                rectangle[i] /= max;
            SDL_RenderDrawMatrix( rpRenderer, position,0,0,0,0, rectangle,Nx,Ny,
                true /*drawAxis*/, "Reconstructed Object" );
        }

        delete[] originalRectangle;
        delete[] rectangle;
    }


} // namespace imresh
} // namespace test

