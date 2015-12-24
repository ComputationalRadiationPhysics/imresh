#pragma once

#include <cmath>  // fmin, fmax

#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif


namespace imresh {
namespace colors {


/**
 * Maps hue(H), saturation(S), value(V) to red, green, blue
 *
 * @verbatim
 *                                                  ^
 *                                               V _|         _______
 *                                       ^          |        /       \
 *                                     S |          |       /         \
 *                                       v  V(1-S) _|______/   Blue    \
 *                                ^                 |
 *                             V _|   ______        +------------------> Hue
 *                                |  /      \       0  1  2  3  4  5  6  * Pi/3
 *                                | /        \
 *          ^             V(1-S) _|/   Green  \______
 *       V _|___            ___   |
 *          |   \   Red    /      +------------------> Hue
 *          |    \        /       0  1  2  3  4  5  6  * Pi/3
 *  V(1-S) _|     \______/
 *          |
 *          +------------------> Hue
 *          0  1  2  3  4  5  6  * Pi/3
 * @endverbatim
 *
 * @param[in] hue assumed to be in [0,2*Pi]. 0: red, 2*Pi/3: green, 4*Pi/4: blue
 *            The hue cycles through all colors.
 * @param[in] saturation assumed to be in [0,1]. This value affects the range
 *            of possible colors (when varying the other 2 parameters).
 *            If 0, then only black will be returned.
 *            If 1, then the full color spectrum can be assumend.
 * @param[in] value assumed to be in [0,1]. For fixed hue and saturation this
 *            basically determines the how many white/light colors we are able
 *            to reach. It's like a top offset. For v=0 everything is black,
 *            but for v=1 we can get the full spectrum, including pure white.
 * @param[out] red will be in [0,1]
 * @param[out] green will be in [0,1]
 * @param[out] blue will be in [0,1]
 **/
void hsvToRgb
( const float hue, const float saturation, const float value,
  float * const red, float * const green, float * const blue );

/**
 * Maps hue, saturation and lightes not red, green, blue
 *
 * The main advantage in contrast to HSV is, that the luminosity will
 * go from black/dark (0) to white/bright (1) instead of the value which
 * goes from black/dark (0) to maximally colorful (1) which would
 * correspond to luminosity = 0.5
 *
 * @param[in] hue assumed to be in [0,2*Pi]. 0: red, 2*Pi/3: green, 4*Pi/4: blue
 *            The hue cycles through all colors.
 * @param[in] saturation assumed to be in [0,1]. This value affects the range
 *            of possible colors (when varying the other 2 parameters).
 *            If 0, then only black will be returned.
 *            If 1, then the full color spectrum can be assumed.
 * @param[in] luminosity assumed to be in [0,1]
 * @param[out] red will be in [0,1]
 * @param[out] green will be in [0,1]
 * @param[out] blue will be in [0,1]
 **/
void hslToRgb
( const float hue, const float saturation, const float luminosity,
  float * const red, float * const green, float * const blue );


} // namespace colors
} // namespace imresh
