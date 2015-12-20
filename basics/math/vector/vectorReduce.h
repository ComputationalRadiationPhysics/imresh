#pragma once

#include <algorithm>  // max


namespace imresh {
namespace math {
namespace vector {


/**
 * Calculate the maximum absolute difference between to arrays
 *
 * Useful for comparing two vectors of floating point numbers
 **/
template<class T>
T vectorMaxAbsDiff( T * const rData1, T * const rData2, const unsigned rnData );


} // namespace vector
} // namespace math
} // namespace imresh
