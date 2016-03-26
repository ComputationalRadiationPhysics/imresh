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


#include "createAtomCluster.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>  // srand, RAND_MAX, rand
#include <cmath>    // fmin, sqrtf, max
#include "libs/gaussian.hpp"


namespace examples
{
namespace createTestData
{


    #define SCALE_EXAMPLE_IMAGE 0


    float * createAtomCluster
    (
        unsigned int const Nx,
        unsigned int const Ny
    )
    {
        assert( Nx > 0 and Ny );

        auto nElements = Nx * Ny;
        auto data = new float[nElements];

        /* Add random background noise and blur it, so that it isn't pixelwise */
        srand(4628941);
        const float noiseAmplitude = 0.00;
        for ( unsigned i = 0; i < nElements; ++i )
            data[i] = 0.7*noiseAmplitude * rand() / (float) RAND_MAX;
        imresh::libs::gaussianBlur( data, Nx, Ny, 1.5 /*sigma in pixels*/ );
        /* add more fine grained noise in a second step */
        for ( unsigned i = 0; i < nElements; ++i )
            data[i] += 0.3*noiseAmplitude * rand() / (float) RAND_MAX;

        /* choose a radious, so that the atom cluster will fit into the image
         * and will fill it pretty well */
        #if SCALE_EXAMPLE_IMAGE == 1
            const float atomRadius = fmin( 0.05f*Nx, 0.01f*Ny );
        #else
            const float atomRadius = 1.6;
        #endif
        #if not defined(NDEBUG) and defined(IMRESH_DEBUG)
            std::cout << "atomRadius = "<<atomRadius<<" px\n";
        #endif
        /* The centers are given as offsets in multiplies of 2*atomradius
         * The virtual position is at ix=iy=0 at first */
        std::vector< std::vector<float> > atomCenters = {
             { 0.45f*Nx/(2*atomRadius), 0.61f*Ny/(2*atomRadius) } /*0*/
            ,{ 0.10f, 0.90f } /*1*/   /*                            */
            ,{ 1.00f,-0.10f } /*2*/   /*       gh                   */
            ,{ 0.70f,-0.70f } /*3*/   /*      df i                  */
            ,{ 0.70f,-0.70f } /*4*/   /*      ce  j k               */
            ,{ 0.94f, 0.15f } /*5*/   /*       b                    */
            ,{-0.70f, 0.70f } /*6*/   /*   1 2 7 8 a                */
            ,{-0.70f, 0.70f } /*7*/   /*   0  3 6 9                 */
            ,{ 0.94f, 0.14f } /*8*/   /*       4 5                  */
            ,{ 0.75f,-0.70f } /*9*/
            ,{ 0.00f,+1.00f } /*a*/
            ,{-1.80f, 0.50f } /*b*/
            ,{-0.70f, 0.70f } /*c*/
            ,{ 0.10f, 0.95f } /*d*/
            ,{ 0.70f,-0.70f } /*e*/
            ,{ 0.10f, 0.95f } /*f*/
            ,{-0.25f, 0.90f } /*g*/
            ,{ 0.90f, 0.14f } /*h*/
            ,{ 0.20f,-0.90f } /*i*/
            ,{ 0.60f,-0.70f } /*j*/
            ,{ 0.65f,-0.60f } /*k*/
            /* second cluster to the lower left from the first */
            ,{-6.00f,-25.0f } /*0*/   /*   ??????????               */
            ,{-0.10f, 0.90f } /*1*/   /*   ??????????               */
            ,{-0.70f, 0.95f } /*2*/   /*   ??????????               */
            ,{ 0.40f, 0.80f } /*3*/   /*   ??????????               */
            ,{ 0.25f, 0.90f } /*4*/   /*   ??????????               */
            ,{ 0.25f, 0.90f } /*5*/   /*   ??????????               */
            ,{ 0.25f, 0.90f } /*6*/   /*   ??????????               */
            ,{-0.60f, 0.90f } /*7*/   /*   ??????????               */
            ,{-0.25f,-0.90f } /*8*/
            ,{-0.25f,-0.90f } /*9*/
            ,{-0.25f,-0.90f } /*a*/
            ,{-0.25f,-1.10f } /*b*/
            ,{ 0.20f, 3.50f } /*c*/
            ,{-0.05f, 1.00f } /*d*/
            ,{-0.15f, 0.90f } /*e*/
        };

        /* spherical intensity function */
        auto f = []( float r ) { return std::abs(r) < 1.0f ? 1.0f - pow(r,6)
                                                           : 0.0f; };
        float x = 0;
        float y = 0;
        for ( auto r : atomCenters )
        {
            x += r[0] * 2*atomRadius;
            y += r[1] * 2*atomRadius;

            int ix0 = std::max( (int) 0   , (int) floor(x-atomRadius)-1 );
            int ix1 = std::min( (int) Nx-1, (int) ceil (x+atomRadius)+1 );
            int iy0 = std::max( (int) 0   , (int) floor(y-atomRadius)-1 );
            int iy1 = std::min( (int) Ny-1, (int) ceil (y+atomRadius)+1 );

            for ( int ix = ix0; ix < ix1; ++ix )
            for ( int iy = iy0; iy < iy1; ++iy )
            {
                data[iy*Nx+ix] += (1-noiseAmplitude) * f( sqrt(pow(ix-x,2)
                                + pow(iy-y,2)) / atomRadius );
                data[iy*Nx+ix] = std::min( 1.0f, data[iy*Nx+ix] );
            }
        }

        return data;
    }


} // namespace createTestData
} // namespace examples
