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


#include "sdlcommon/sdlcommon.h"
#include "testVectorIndex.h"
#include "testDiffractionIntensity.h"
#include "testHybridInputOutput.h"


int main(void)
{
    /* Tests which don't need to show graphics */
    imresh::test::testVectorIndex();
    std::cout << "testVectorIndex [OK]\n";


    using namespace imresh::sdlcommon;
    using namespace imresh::test;

    SDL_Window   * pWindow;
    SDL_Renderer * pRenderer;

    /* Initialize SDL Context */
    SDL_CHECK( SDL_Init( SDL_INIT_VIDEO ) )

    pWindow = SDL_CreateWindow( "Output",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        1024, 640, SDL_WINDOW_SHOWN );
    SDL_CHECK( pWindow );

    pRenderer = SDL_CreateRenderer( pWindow, -1, SDL_RENDERER_ACCELERATED );
    SDL_CHECK( pRenderer );

    SDL_SetRenderDrawColor( pRenderer, 255,255,255,255 );
    SDL_RenderClear( pRenderer );
    SDL_RenderPresent( pRenderer );

    /* Wait for key to quit */
    int mainProgrammRunning = 1;
    int currentFrame = 0;
    bool drawNext = true;
    while (mainProgrammRunning)
    {
        /* Handle Keyboard and Mouse events */
        SDL_Event event;
        while ( SDL_PollEvent(&event) )
        {
            mainProgrammRunning &= not SDL_basicControl(event,pWindow,pRenderer);
            SDL_SetRenderDrawColor( pRenderer, 128,0,0,255 );
            drawNext |= SDL_animControl( event );
            if ( drawNext )
            {
                drawNext = false;

                SDL_SetRenderDrawColor( pRenderer, 255,255,255,255 );
                SDL_RenderClear( pRenderer );

                SDL_SetRenderDrawColor( pRenderer, 0,0,0,255 );
                switch ( currentFrame % 2 )
                {
                    case 0:
                        testHybridInputOutput(pRenderer);
                        break;
                    case 1:
                        testDiffractionIntensity(pRenderer);
                        break;
                    /*
                    case 2: imresh::test::testGaussian(pRenderer);   break;
                    case 3: imresh::test::testGaussian2d(pRenderer); break;
                    case 4: imresh::test::testDft(pRenderer);        break;
                    case 5: imresh::test::testFftw(pRenderer);       break;
                    case 6: imresh::test::testFftw2d(pRenderer);     break;
                    */
                    default: break;
                }
                SDL_RenderPresent( pRenderer );

                ++currentFrame;
            }
        }
        SDL_Delay(50 /*ms*/);
    }

	SDL_DestroyWindow( pWindow );
	SDL_Quit();
    return 0;
}
