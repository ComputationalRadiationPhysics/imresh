
#include <cstdlib>  // srand, RAND_MAX, rand
#include <cmath>    // fmin, sqrtf
#include <vector>
#include "math/image/gaussian.h"


namespace imresh {
namespace examples {


//createLoadFromFile ???

//createCheckerboard2d
//createCheckerboard2d
//createNoise2d
//createSphere2d
//createHalfSphere2d
//createCone2d
//createHalfCone2d

//createCheckerboard3d
//createCheckerboard3d
//createNoise3d
//createSphere3d
//createHalfSphere3d
//createCone3d
//createHalfCone3d



float * createAtomCluster( const int Nx, const int Ny )
{
    using namespace imresh::math::image;

    float * data = new float[Nx*Ny];

    /* Add random background noise and blur it, so that it isn't pixelwise */
    srand(4628941);
    const float noiseAmplitude = 0.3;
    for ( int i = 0; i < Nx*Ny; ++i )
        data[i] = 0.7*noiseAmplitude * rand() / (float) RAND_MAX;
    gaussianBlur( data, Nx, Ny, 1.5 /*sigma in pixels*/ );
    /* add more fine grained noise in a second step */
    for ( int i = 0; i < Nx*Ny; ++i )
        data[i] += 0.3*noiseAmplitude * rand() / (float) RAND_MAX;

    /* choose a radious, so that the atom cluster will fit into the image and
     * will fill it pretty well */
    const float atomRadius = fmin( 0.05f*Nx, 0.01f*Ny );
    std::cout << "atomRadius = "<<atomRadius<<" px\n";
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
        #if false
        ,{ 0.65f,-0.60f } /*k*/
        #endif
        /* second cluster to the lower left from the first */
        ,{-6.00f,-25.0f }
        ,{-0.10f, 0.90f }
        ,{-0.70f, 0.95f }
        ,{ 0.40f, 0.80f }
        ,{ 0.25f, 0.90f }
        ,{ 0.25f, 0.90f }
        ,{ 0.25f, 0.90f }
        ,{-0.60f, 0.90f }
        ,{-0.25f,-0.90f }
        ,{-0.25f,-0.90f }
        ,{-0.25f,-0.90f }
        ,{-0.25f,-1.10f }
        ,{ 0.20f, 3.50f }
        ,{-0.05f, 1.00f }
        ,{-0.15f, 0.90f }
    };

    /* spherical intensity function */
    auto f = []( float r ) { return std::abs(r) < 1.0f ? 1.0f-pow(r,6) : 0.0f; };
    float x = 0;
    float y = 0;
    for ( auto r : atomCenters )
    {
        x += r[0] * 2*atomRadius;
        y += r[1] * 2*atomRadius;

        int ix0 = std::max( 0   , (int) floor(x-atomRadius)-1 );
        int ix1 = std::min( Nx-1, (int) ceil (x+atomRadius)+1 );
        int iy0 = std::max( 0   , (int) floor(y-atomRadius)-1 );
        int iy1 = std::min( Ny-1, (int) ceil (y+atomRadius)+1 );

        for ( int ix = ix0; ix < ix1; ++ix )
        for ( int iy = iy0; iy < iy1; ++iy )
        {
            data[iy*Nx+ix] += (1-noiseAmplitude)*f( sqrt(pow(ix-x,2) + pow(iy-y,2))/atomRadius );
            data[iy*Nx+ix] = std::min( 1.0f, data[iy*Nx+ix] );
        }
    }

    return data;
}


} // namespace examples
} // namespace imresh
