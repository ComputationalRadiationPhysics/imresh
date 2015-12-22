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

template<class T>
T vectorMax( T * const rData, const unsigned rnData );

template<class T>
T vectorMin( T * const rData, const unsigned rnData );


} // namespace vector
} // namespace math
} // namespace imresh
