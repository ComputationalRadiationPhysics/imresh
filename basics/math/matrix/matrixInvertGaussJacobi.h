#pragma once

#include <cassert>


namespace imresh {
namespace math {
namespace matrix {


/**
 * Calculates the invere matrix of A and stores the result in A
 *
 * @param[in]  A matrix to invert
 * @param[in]  N number of rows and colums of A (square matrix assumend).
 *             the memory location A points to needs to be at least N*N*
 *             sizeof(T_PREC) bytes large
 * @param[out] A the inverted result
 **/
template<class T_PREC>
void matrixInvertGaussJacobi( T_PREC * const A, const unsigned N );


} // namespace matrix
} // namespace math
} // namespace imresh
