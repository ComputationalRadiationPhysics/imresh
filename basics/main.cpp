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
 - show how large the kernel must be, so that \int_-infty^? 255 gaus(x) dx < 0.5
 - kernel == vector of weights, meaning this i the same as newtonCotes !!
 - similarily show results of different gaussian blurs in 1D

 - make ft with continuous spectra work (bzw. DFT), also show like above

 - implement gaussian blur in 2D
 - generate some kind of test data -> random, checkerboard, circle, ...
 - show effect of gaussian blur in 2D
 - show how gaussian blur can first be applied in x, then in y-direction, plot both steps
 - implement plot2D with macroPixel width,height and gray values
 - bonus: try it with real image :3 -> see C++ program from back then, which display iriya no sora, ufo no natsu bmp :333, or was that pascal :X

*/




#include "gaussian.h"
#include <cstdlib> // srand, rand, RAND_MAX

void testGaussianBlurVector
( SDL_Renderer * rpRenderer, SDL_Rect rect, float * data, int nData,
  const float sigma, const char * title )
{
    SDL_RenderDrawHistogram(rpRenderer, rect, 0,0,0,0,
        data,nData, 0/*binWidth*/,false/*fill*/, true/*drawAxis*/, title );
    SDL_RenderDrawArrow( rpRenderer, rect.x+1.1*rect.w,rect.y+rect.h/2,
                                     rect.x+1.3*rect.w,rect.y+rect.h/2 );
    rect.x += 1.5*rect.w;

    gaussianBlur( data, nData, sigma );

    char title2[128];
    sprintf( title2,"G(s=%0.1f)*%s",sigma,title );
    SDL_RenderDrawHistogram(rpRenderer, rect, 0,0,0,0,
        data,nData, 0/*binWidth*/,false/*fill*/, true/*drawAxis*/, title2 );
}

void testGaussian( SDL_Renderer * rpRenderer )
{
    srand(165158631);
    SDL_Rect rect = { 40,40,200,80 };

    const int nData = 50;
    float data[nData];

    /* Try different data sets */
    for ( int i = 0; i < nData; ++i )
        data[i] = 255*rand()/(double)RAND_MAX;
    testGaussianBlurVector( rpRenderer,rect,data,nData, 1.0, "Random" );
    rect.y += 100;
    for ( int i = 0; i < nData; ++i )
        data[i] = 255*rand()/(double)RAND_MAX;
    testGaussianBlurVector( rpRenderer,rect,data,nData, 2.0, "Random" );
    rect.y += 100;
    for ( int i = 0; i < nData; ++i )
        data[i] = 255*rand()/(double)RAND_MAX;
    testGaussianBlurVector( rpRenderer,rect,data,nData, 4.0, "Random" );
    rect.y += 100;

    for ( int i = 0; i < nData; ++i )
        data[i] = i > nData/2 ? 1 : 0;
    testGaussianBlurVector( rpRenderer,rect,data,nData, 1.0, "Step" );
    rect.y += 100;
    for ( int i = 0; i < nData; ++i )
        data[i] = i > nData/2 ? 1 : 0;
    testGaussianBlurVector( rpRenderer,rect,data,nData, 4.0, "Step" );
    rect.y += 100;

    {
    const int nData2 = 100;
    float data2[nData2];
    float sigma = 8.0;
    float a =  1.0/( sqrt(2.0*M_PI)*sigma );
    float b = -1.0/( 2.0*sigma*sigma );
    for ( int i = 0; i < nData2; ++i )
        data2[i] = a*exp( (i-nData2/2)*(i-nData2/2)*b );
    char title[64];
    sprintf(title,"G(s=%.2f)",sigma);

    testGaussianBlurVector( rpRenderer,rect,data2,nData2, sigma, title );
    rect.y += 100;
    }
}

void testGaussianConvergence( SDL_Renderer * rpRenderer )
{
    SDL_Rect rect = { 40,40,200,80 };

    const int nData = 50;
    float data[nData];

    /* Delta Peak */
    for ( int i = 0; i < nData; ++i )
        data[i] = 0;
    data[nData/2] = 1;

    testGaussianBlurVector( rpRenderer,rect,data,nData, 1.0, "1" );
    rect.y += 100;
    testGaussianBlurVector( rpRenderer,rect,data,nData, 1.0, "1" );
    rect.y += 100;
}


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
        800, 700, SDL_WINDOW_SHOWN );
    SDL_CHECK( pWindow );

    pRenderer = SDL_CreateRenderer( pWindow, -1, SDL_RENDERER_ACCELERATED );
    SDL_CHECK( pRenderer );

    SDL_SetRenderDrawColor( pRenderer, 255,255,255,255 );
    SDL_RenderClear( pRenderer );
    SDL_RenderPresent( pRenderer );

    //testSdlPlot(pRenderer);
    /* Do and plot tests */
    SDL_SetRenderDrawColor( pRenderer, 0,0,0,255 );
    //testDcft(pRenderer);
    testGaussian(pRenderer);


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
