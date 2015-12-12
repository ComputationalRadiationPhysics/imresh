/*
file=main; rm $file.exe; g++ -g -std=c++11 -Wall -Wextra -Wshadow -Wno-unused-parameter $file.cpp -o $file.exe $(sdl2-config --cflags --libs) -l SDL2_image -l SDL2_ttf; ./$file.exe
*/

#include "dcft.cpp"
#include "sdlcommon.h"
#include "sdlplot.h"

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


/*
todo: histogram, if smaller than zero, then plot bar downwards ...
plot x-axis in middle for histogram, instead of down below!

Instead of simple histplots, draw:

                           +------------+
                           | cos coeffs |
        +---------+        +------------+
        |functions|  ->    +------------+
        +---------+        | sin coeffs |
                           +------------+

 - show how large the kernel must be, so that \int_-infty^? 255 gaus(x) dx < 0.5
 - kernel == vector of weights, meaning this i the same as newtonCotes !!
 - similarily show results of different gaussian blurs in 1D

 - make ft with continuous spectra work (bzw. DFT), also show like above
 - Re and Im, maybe even in same plot :3 -> plot colors and plot labels
 - make title work, and or Plot Legend

 - implement gaussian blur in 2D
 - generate some kind of test data -> random, checkerboard, circle, ...
 - show effect of gaussian blur in 2D
 - show how gaussian blur can first be applied in x, then in y-direction, plot both steps
 - implement plot2D with macroPixel width,height and gray values
 - bonus: try it with real image :3 -> see C++ program from back then, which display iriya no sora, ufo no natsu bmp :333, or was that pascal :X

*/








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
        800, 600, SDL_WINDOW_SHOWN );
    SDL_CHECK( pWindow );

    pRenderer = SDL_CreateRenderer( pWindow, -1, SDL_RENDERER_ACCELERATED );
    SDL_CHECK( pRenderer );

    SDL_SetRenderDrawColor( pRenderer, 255,255,255,255 );
    SDL_RenderClear( pRenderer );
    SDL_RenderPresent( pRenderer );

    //testSdlPlot(pRenderer);
    /* Do and plot tests */
    SDL_SetRenderDrawColor( pRenderer, 0,0,0,255 );
    testDcft(pRenderer);


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
