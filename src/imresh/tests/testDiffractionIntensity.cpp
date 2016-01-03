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


#include "testDiffractionIntensity.h"


namespace imresh
{
namespace test
{


    void testDiffractionIntensity
    ( SDL_Renderer * const & rpRenderer )
    {
        std::vector<unsigned> imageSize = {20,20};
        const unsigned & Nx = imageSize[1];
        const unsigned & Ny = imageSize[0];
        const unsigned w  = 4;
        const unsigned h  = 8;

        /* initialize rectangle */
        float * rectangle = new float[Nx*Ny];
        memset( rectangle,0, Nx*Ny*sizeof( rectangle[0] ) );
        for ( unsigned ix = Nx/2-w; ix <= Nx/2+w; ++ix )
        for ( unsigned iy = Ny/2-h; iy <= Ny/2+h; ++iy )
            rectangle[iy*Nx+ix] = 1.0f;

        /* display it */
        using namespace sdlcommon;
        SDL_Rect position = { 40,30,200,200 };
        SDL_RenderDrawMatrix( rpRenderer, position,0,0,0,0, rectangle,Nx,Ny,
            true /*drawAxis*/, "Rectangular Opening" );
        SDL_RenderDrawArrow( rpRenderer,
            position.x + 1.1*position.w, position.y + position.h/2,
            position.x + 1.4*position.w, position.y + position.h/2 );
        position.x += 1.5*position.w;

        /* convert with slightly less performant method */
        fftw_complex * F = fftw_alloc_complex( Nx*Ny );
        {
            for ( unsigned i = 0; i < Nx*Ny; ++i )
            {
                F[i][0] = rectangle[i];
                F[i][1] = 0;
            }
            /* fourier transform the original image */
            fftw_plan planRectToDiff = fftw_plan_dft_2d( Nx,Ny, F,F,
                FFTW_FORWARD, FFTW_ESTIMATE );
            fftw_execute( planRectToDiff );
            fftw_destroy_plan( planRectToDiff );
            /* strip fourier transformed real image of it's phase (measurement) */
            for ( unsigned i = 0; i < Nx*Ny; ++i )
            {
                const float & re = F[i][0]; /* Re */
                const float & im = F[i][1]; /* Im */
                F[i][0] = sqrtf( re*re + im*im );
                F[i][1] = 0;
            }
        }

        /* convert to diffraction pattern */
        algorithms::diffractionIntensity( rectangle, imageSize );

        /* compare diffractionIntensity result with slower method */
        {
            bool foundInequality = false;
            for ( unsigned i = 0; i < Nx*Ny; ++i )
            {
                float max = fmax( F[i][0] , rectangle[i] );
                if ( max == 0 ) max = 1.0;
                bool equal = fabs( F[i][0] - rectangle[i] ) / max
                             < 10*FLT_EPSILON;
                if ( not equal )
                    std::cout << "i = " << i << " is not equal. ( "
                    << F[i][0] << " != " << rectangle[i] << ")\n";
                foundInequality |= not equal;
            }
            assert( not foundInequality );
        }

        /* do some sanity checks on the diffraction pattern */
        {
            std::cout << std::setprecision(8) ;
            bool foundInequality = false;
            for ( unsigned ix = 1; ix < Nx; ++ix )
            for ( unsigned iy = 1; iy < Ny; ++iy )
            {
                bool equalX = fabs( rectangle[ iy*Nx + ix ]
                                  - rectangle[ iy*Nx + Nx -ix ] )
                              < 10*FLT_EPSILON;
                if ( not equalX )
                    std::cout << "ix = " << ix << ", iy = " << iy
                    << " ( " << rectangle[ iy*Nx + ix ] << " != "
                    << rectangle[ iy*Nx + Nx -ix ] << ")\n";

                bool equalY = fabs( rectangle[ iy*Nx + ix ]
                                  - rectangle[ ( Ny - iy )*Nx + ix ] )
                              < 20*FLT_EPSILON;
                if ( not equalY )
                    std::cout << "ix = " << ix << ", iy = " << iy
                    << " ( " << rectangle[ iy*Nx + ix ] << " != "
                    << rectangle[ (Ny-iy)*Nx + ix ] << ")\n";

                foundInequality |= not equalX;
                foundInequality |= not equalY;
            }
            //assert( not foundInequality );
        }

        /* compare with analytical solution. Not that easy, because it is
         * a discrete fourier transform */
        /*
        auto sinc = []( double x ) { return fabs(x) < FLT_EPSILON ?
            cos(x) : sin(x) / x; };
        for ( unsigned iy = 0; iy < 1; ++iy )
        for ( unsigned ix = 0; ix < Nx; ++ix )
        {
            float omegaX = ix <= Nx/2 ? ix : ix-Nx;
            float omegaY = iy <= Ny/2 ? iy : iy-Ny;
            std::cout << "ix="<<ix<<",iy="<<iy<<": "<< rectangle[iy*Nx+ix] /
            sinc(omegaX)*sinc(omegaY) << "\n";
        }
        */


        /* display pattern */
        //for ( unsigned i = 0; i < Nx*Ny; ++i )
        //    rectangle[i] = logf( 1+rectangle[i] );
        float max = 0;
        for ( unsigned i = 0; i < Nx*Ny; ++i )
            max = std::max( max, rectangle[i] );
        for ( unsigned i = 0; i < Nx*Ny; ++i )
            rectangle[i] /= max;
        SDL_RenderDrawMatrix( rpRenderer, position,0,0,0,0, rectangle,Nx,Ny,
            true /*drawAxis*/, "Diffraction Intensity" );

        fftw_free( F );
        delete[] rectangle;
    }


} // namespace imresh
} // namespace test

