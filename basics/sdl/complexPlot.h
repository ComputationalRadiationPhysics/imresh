#pragma once

#include <cmath>    // log10
#include <complex>
#include <SDL.h>
#include <fftw3.h>
#include "sdlplot.h"
#include "colors/conversions.h"


namespace sdlcommon {


/**
 * Uses domain coloring to plot a complex valued matrix
 *
 * The matrix could e.g. contain evaluations of a complex function.
 *
 * @param[in] logPlot if true, then instead of the magnitude of the complex
 *            number log(|z|) will be plotted.
 * @param[in] swapQuadrants true: rows and columns will be shifted by half
 *            width thereby centering the shortest wavelengths instead of
 *            those being at the corners
 * @param[in] colorFunction 1:HSL (H=arg(z), S=1, L=|z|)
 *                          2:HSV
 *                          3:
 * @see SDL_RenderDrawMatrix for other parameters
 **/
void SDL_RenderDrawComplexMatrix
(
  SDL_Renderer * const rpRenderer, const SDL_Rect & rAxes,
  const float x0, const float x1, const float y0, const float y1,
  fftw_complex * const values, const unsigned nValuesX, const unsigned nValuesY,
  const bool drawAxis, const char * const title,
  const bool logPlot = true,
  const bool swapQuadrants = false, const int colorFunction = 2
);


} // namespace sdlcommon
