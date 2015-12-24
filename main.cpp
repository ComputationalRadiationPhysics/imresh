#include <SDL.h>
#include "sdl/sdlcommon.h"

#include "tests/testAtomCluster.cpp"


int main(void)
{
    using namespace sdlcommon;

    SDL_Window   * pWindow;
    SDL_Renderer * pRenderer;

    /* Initialize SDL Context */
    SDL_CHECK( SDL_Init( SDL_INIT_VIDEO ) )
    pWindow = SDL_CreateWindow( "Output",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        500, 700, SDL_WINDOW_SHOWN );
    SDL_CHECK( pWindow );

    pRenderer = SDL_CreateRenderer( pWindow, -1, SDL_RENDERER_ACCELERATED );
    SDL_CHECK( pRenderer );

    SDL_SetRenderDrawColor( pRenderer, 255,255,255,255 );
    SDL_RenderClear( pRenderer );
    SDL_RenderPresent( pRenderer );
    SDL_SetRenderDrawColor( pRenderer, 0,0,0,255 );


    /* Do and plot tests */
    testAtomCluster(pRenderer);

    /* Wait for key to quit */
    int mainProgrammRunning = 1;
    int renderTouched = 1;
    while (mainProgrammRunning)
    {
        SDL_Event event;
        while ( SDL_PollEvent(&event) )
            mainProgrammRunning &= not SDL_basicControl(event,pWindow,pRenderer);
            SDL_SetRenderDrawColor( pRenderer, 128,0,0,255 );
        if ( renderTouched )
        {
            renderTouched = 0;
            SDL_RenderPresent( pRenderer );
        }
        SDL_Delay(50 /*ms*/);
    }
	SDL_DestroyWindow( pWindow );
	SDL_Quit();

    return 0;
}
