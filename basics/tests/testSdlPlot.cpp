
#include <SDL.h>
#include "sdl/sdlplot.h"


namespace sdlcommon {
namespace test {


int testSdlPlot( SDL_Renderer * rpRenderer )
{
    SDL_SetRenderDrawColor( rpRenderer, 0,0,0,255 );
    const int nAxis = 5;
    SDL_Rect axis[nAxis];
    for ( int i=0; i<nAxis; ++i )
        axis[i] = SDL_Rect{ 40+i*150,120, 100,100 /*w,h*/};
    auto f = [](float x){return sin(x);};
    std::cout << "Print axes at " << axis[0] << "\n";
    SDL_RenderDrawAxes( rpRenderer, axis[0], 0,10     ,0,10       );
    SDL_RenderDrawAxes( rpRenderer, axis[1], 1,1285   , 1,1285    );
    SDL_RenderDrawAxes( rpRenderer, axis[2],-0.05,0.07,-0.05,0.07 );
    SDL_RenderDrawAxes( rpRenderer, axis[3],-1e-7,0   ,-1e-7,0    );
    SDL_RenderDrawAxes( rpRenderer, axis[4],0,9.9     ,0,9.9      );
    for ( int i=0; i<nAxis; ++i )
        axis[i].y += 125;
    SDL_RenderDrawFunction( rpRenderer, axis[0], 1.3,23.7 ,0,0, f, true );
    SDL_RenderDrawFunction( rpRenderer, axis[1], 1,1285   ,0,0, f, true );
    SDL_RenderDrawFunction( rpRenderer, axis[2],-0.05,0.07,0,0, f, true );
    SDL_RenderDrawFunction( rpRenderer, axis[3],-1e-7,0   ,0,0, f, true );

    return 0;
}


} // namespace test
} // namespace sdlcommon
