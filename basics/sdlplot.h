#pragma once


#include <iostream>
#include <SDL.h>
#include "sdlcommon.h"
#include <cmath>    // log10
#include <cstddef>  // NULL

class SDL_PlotFonts {
private:
    static SDL_PlotFonts * mInstance;

    /* comparing the output with
     * https://github.com/chrissimpkins/Hack/raw/master/img/hack-waterfall.png
     * it seems that fontsize 8pt corresponds to the value 9 inside SDL_ttf
     * Bug ??? */
    const int mTickFontSize  = 9;
    const int mLabelFontSize = 10;

public:
    TTF_Font * tickFont;
    TTF_Font * labelFont;

    SDL_PlotFonts()
    {
        if ( not TTF_WasInit() )
            TTF_Init();

        tickFont = TTF_OpenFont( "Hack/Hack-Regular.ttf", mTickFontSize );
        CHECK_TTF_PTR( tickFont );
        labelFont = TTF_OpenFont( "Hack/Hack-Regular.ttf", mLabelFontSize );
        CHECK_TTF_PTR( labelFont );
    }
    ~SDL_PlotFonts()
    {
        TTF_CloseFont(tickFont);
        TTF_CloseFont(labelFont);
    }
    static SDL_PlotFonts * instance(void)
    {
        if ( mInstance == NULL )
            mInstance = new SDL_PlotFonts();
        return mInstance;
    }
};
SDL_PlotFonts * SDL_PlotFonts::mInstance = 0;

/**
 * Renders a text string to the given renderer
 *
 * @param[in]  rpRenderer renderer to use
 * @param[in]  rFont font to use
 * @param[in]  rString string to print
 * @param[in]  rTarget location where to print. If w or h are 0 they are
 *             filled in to match the width and height of the rendered string,
 *             else the texture will be rendered scaled
 * @param[in]  rXAlign 0 if normal left alignment, 1 if central alignment,
 *             2 if right alignment, meaning right side of string to render
 *             touches rTarget.x,y
 * @param[in]  rYAlign vertical alignment, @see rXAlign
 * @param[out] rTarget will be changed if necessary for align or if h or w
 *             were set to 0.
 **/
int SDL_drawString
( SDL_Renderer * rpRenderer, TTF_Font * rFont, const char * rString,
  SDL_Rect * rTarget, int rXAlign, int rYAlign );

/**
 * Choose ticks in such a way, that at least 10 ticks are shown
 **/
float chooseTickSpacing(float a, float b);

int printPrettyFloat( char * rString, float rValue );

/**
 * Draw a vertical axis for a plot labeled with values
 *
 * If automatically chooses ticks in such a way, that at least 10 ticks are
 * shown.
 *
 * @param[in] rpRenderer renderer to draw on
 * @param[in] x position of origo, meaning the lower bound of the axis
 * @param[in] y position of origo, note that in the SDL coordinate system
 *            used here larger y values mean more to the bottom in the window
 * @param[in] h height of the axis in pixels
 * @param[in] y0 lower value bound of the axis, used for drawing labels
 * @param[in] y1 upper value bound of the axis
 **/
int SDL_RenderDrawAxes
( SDL_Renderer * rpRenderer, const SDL_Rect & rAxes,
  float x0, float x1, float y0, float y1 );

/**
 * Wrapper for SDL_RenderDrawAxes using SDL_Rect instead of SDL_Rect*
 **/
int SDL_RenderDrawAxes
( SDL_Renderer * rpRenderer, const SDL_Rect * rAxes,
  float x0, float x1, float y0, float y1 )
{ return SDL_RenderDrawAxes(rpRenderer,*rAxes,x0,x1,y0,y1); }


#include <cfloat>  // FLT_MAX, FLT_MIN

template<class T_FUNC>
int SDL_RenderDrawFunction
( SDL_Renderer * rpRenderer, const SDL_Rect & rAxes,
  float x0, float x1, float y0, float y1, T_FUNC f, bool drawAxis = false );

int SDL_RenderDrawHistogram
( SDL_Renderer * rpRenderer, const SDL_Rect & rAxes,
  float x0, float x1, float y0, float y1,
  float * values, int nValues,
  int binWidth = 0,  bool fill = false,
  bool drawAxis = false, const char * title = "" );





#include "sdlplot.cpp"
