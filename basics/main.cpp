/*
file=main; rm $file.exe; g++ -g -std=c++11 -Wall -Wextra -Wshadow -Wno-unused-parameter $file.cpp -o $file.exe $(sdl2-config --cflags --libs) -l SDL2_image -l SDL2_ttf; ./$file.exe
*/

#include "math/fouriertransform/dcft.h"
#include "math/fouriertransform/dft.h"
#include "sdl/sdlcommon.h"
#include "sdl/sdlplot.h"

using namespace imresh::math::fouriertransform;

/*
 - generate some kind of test data -> random, checkerboard, circle, ...
 - bonus: try it with real image :3 -> see C++ program from back then, which display iriya no sora, ufo no natsu bmp :333, or was that pascal :X
*/

#include "tests/testSdlPlot.cpp"
#include "tests/testDcft.cpp"
#include "tests/testDft.cpp"
#include "tests/testFftw.cpp"
#include "tests/testGaussian.cpp"
#include "tests/testGaussian2d.cpp"
#include "tests/testMatrixInvertGaussJacobi.cpp"

int main(void)
{
    SDL_Window   * pWindow;
    SDL_Renderer * pRenderer;

    /* Initialize SDL Context */
    SDL_CHECK( SDL_Init( SDL_INIT_VIDEO ) )


    const int numdrivers = SDL_GetNumRenderDrivers();
    std::cout << "Render driver count: " << numdrivers << "\n";
    for ( int i = 0; i < numdrivers; i++ )
    {
        SDL_RendererInfo drinfo;
        SDL_GetRenderDriverInfo (i, &drinfo);
        std::cout << "Driver name ("<<i<<"): " << drinfo.name << " flags: ";
        if (drinfo.flags & SDL_RENDERER_SOFTWARE)      printf("Software ");
        if (drinfo.flags & SDL_RENDERER_ACCELERATED)   printf("Accelerated ");
        if (drinfo.flags & SDL_RENDERER_PRESENTVSYNC)  printf("VSync ");
        if (drinfo.flags & SDL_RENDERER_TARGETTEXTURE) printf("Textures ");
        printf("\n");
    }


    pWindow = SDL_CreateWindow( "Output",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        900, 650, SDL_WINDOW_SHOWN );
    SDL_CHECK( pWindow );

    pRenderer = SDL_CreateRenderer( pWindow, -1, SDL_RENDERER_ACCELERATED );
    SDL_CHECK( pRenderer );

    SDL_SetRenderDrawColor( pRenderer, 255,255,255,255 );
    SDL_RenderClear( pRenderer );
    SDL_RenderPresent( pRenderer );

    /* Do and plot tests */
    SDL_SetRenderDrawColor( pRenderer, 0,0,0,255 );
    //sdlcommon::test::testSdlPlot(pRenderer);
    imresh::test::testMatrixInvertGaussJacobi();
    //imresh::test::testDcft(pRenderer);
    //imresh::test::testGaussian(pRenderer);
    //imresh::test::testGaussian2d(pRenderer);
    imresh::test::testDft(pRenderer);
    //imresh::test::testFftw(pRenderer);
    //imresh::test::testFftw2d(pRenderer);


    //SDL_drawLineControl drawControl;

    /* Wait for key to quit */
    int mainProgrammRunning = 1;
    int renderTouched = 1;
    while (mainProgrammRunning)
    {
        /* Handle Keyboard and Mouse events */
        SDL_Event event;
        while ( SDL_PollEvent(&event) )
        {
            mainProgrammRunning &= not SDL_basicControl(event,pWindow,pRenderer);
            SDL_SetRenderDrawColor( pRenderer, 128,0,0,255 );
            //renderTouched |= drawControl(event, pRenderer);
        }

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
