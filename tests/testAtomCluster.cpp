
#include "sdl/sdlplot.h"
#include "examples/createAtomCluster.cpp"

void testAtomCluster( SDL_Renderer * rpRenderer )
{
    using namespace imresh::examples;
    using namespace sdlcommon;

    const int Nx = 200, Ny = 300;
    float * cluster = createAtomCluster( Nx, Ny );

    SDL_Rect axis = { 30,20, Nx,Ny };
    SDL_RenderDrawMatrix( rpRenderer, axis, 0,0,0,0, cluster,Nx,Ny,
                          /*drawAxis*/ true, "Atom Cluster Original" );

    delete[] cluster;
}
