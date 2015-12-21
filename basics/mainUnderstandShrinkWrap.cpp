
#include "sdl/sdlcommon.h"
#include "../examples/createAtomCluster.cpp"


class AnimateShrinkWrap
{

private:
    int mCurrentFrame;

    const unsigned Nx, Ny;
    float * mOriginalImage;
    float * mFTImage;
    float * mFTIntensity;

public:
    AnimateShrinkWrap
    ( float * const rpOriginalData, const unsigned rNx, const unsigned rNy )
     : Nx(rNx), Ny(rNy)
    {
        const unsigned dataSize = Nx*Ny*sizeof(float);
        mOriginalImage = (float*) malloc( dataSize );
        mFTImage       = (float*) malloc( dataSize );
        mFTIntensity   = (float*) malloc( dataSize );
        memcpy( mOriginalImage, rpOriginalData, dataSize );
    }
    ~AnimateShrinkWrap()
    {
        free( mOriginalImage );
        free( mFTImage       );
        free( mFTIntensity   );
    }

    void step( void ) {};

};



int main(void)
{
    using namespace sdlcommon;

    SDL_Window   * pWindow;
    SDL_Renderer * pRenderer;

    /* Initialize SDL Context */
    SDL_CHECK( SDL_Init( SDL_INIT_VIDEO ) )

    pWindow = SDL_CreateWindow( "Understand Shrink-Wrap",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        1024, 640, SDL_WINDOW_SHOWN );
    SDL_CHECK( pWindow );

    pRenderer = SDL_CreateRenderer( pWindow, -1, SDL_RENDERER_ACCELERATED );
    SDL_CHECK( pRenderer );

    SDL_SetRenderDrawColor( pRenderer, 255,255,255,255 );
    SDL_RenderClear( pRenderer );
    SDL_RenderPresent( pRenderer );


    using namespace imresh::examples;
    const unsigned Nx = 200, Ny = 300;
    float * atomCluster = createAtomCluster( Nx, Ny );

    AnimateShrinkWrap animateShrinkWrap( atomCluster, Nx, Ny );
    free(atomCluster);
    animateShrinkWrap.step();


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
