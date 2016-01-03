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
