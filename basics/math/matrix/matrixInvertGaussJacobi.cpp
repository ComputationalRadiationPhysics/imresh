
#include "matrixInvertGaussJacobi.h"


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
void matrixInvertGaussJacobi( T_PREC * const A, const unsigned N )
{
    /* Bring matrix in row echelon form
     * (everything except upper right triangle is 0):
     *
     * iRow = 0 (begins):
     *   ( a00 a01 a02 .. | 1           .. ) |*1/a00  |*a10 |   |*a20 |   ...
     *   ( a10 a11 a12 .. |     1       .. )                v-        |
     *   ( a20 a21 a22 .. |         1   .. )                          v-
     *   ( ..  ..  ..  .. | ..  ..  ..  .. )
     *
     * iRow = 1 (begins):
     *   ( 1   b01 b02 .. |   1/a00           .. )
     *   (     b11 b12 .. |-a10/a00   1       .. )  |*1/b11  |*b21 |   ...
     *   (     b21 b22 .. |-a20/a00       1   .. )                 v-
     *   ( ..  ..  ..  .. | ..        ..  ..  .. )
     *
     * iRow = 2 (begins):
     *   ( 1   b01 b02 .. | 1/a00             .. )
     *   (     1   c12 .. | h10     1/b11     .. )
     *   (         c22 .. | h20  -b21/b11 1   .. )  |*1/c22  |*c32 |   ...
     *   ( ..  ..  ..  .. | ..    ..      ..  .. )                 v-
     *
     *                   ...
     *
     * iRow = N-1 (completed):
     *   ( 1   b01 b02 .. | 1/a00             .. )
     *   (     1   c12 .. | h10   1/b11       .. )
     *   (         1   .. | h20   h21   1/c22 .. )
     *   ( ..  ..  ..  .. | ..    ..    ..    .. )
     *
     * because after the first step the 1st column will be known completely
     * (it's first element is 1, the rest is 0), we can use these known
     * memory locations to store the result of the 1st column of the right
     * half:
     *
     * iRow = 1 (begins):
     *   (   1/a00 | b01 b02 .. )
     *   (-a10/a00 | b11 b12 .. )  |*1/b11, A11=1/b11  |*b21 |   |*b31 |   ...
     *   (-a20/a00 | b21 b22 .. )                            v-        |
     *   (-a30/a00 | b31 b32 .. )                                      v-
     *   ( ..      | ..  ..  .. )
     *
     * iRow = 2 (begins):
     *   ( 1/a00 \  b01      b02 .. )
     *   (        ---------         )
     *   ( h10       1/b11 \ b12 .. )  |*1/b22, A22=1/b22  |*b32 |
     *   ( h20    -b21/b11 | b22 .. )                            v-
     *   ( h30    -b31/b11 | b32 .. )
     *   ( ..      ..      | ..  .. )
     *
     * similarily this is true for the next steps:
     *
     *   ( 1/a00 \ b01     b02     .. )
     *   (        -------             )
     *   ( h10     1/b11 \ b12     .. )
     *   (                -------     )
     *   ( h20     h21     1/c22 \ .. )
     *   ( ..      ..      ..      .. )
     *
     * The steps then actually simplify, because we subtraction of iRow
     * becomes a substraction in the merged matrix! We only have to set
     * the diagonal elements to their correct values after the multiplication
     * of iRow, because else the diagonal elements would be 1 instead of 1/aii
     **/
    for ( unsigned iRow = 0; iRow < N; iRow++ )
    {
        T_PREC * rowSubtrahend = &A[iRow*N];

        /* Divide complete row by diagonal element a_00,a_11,... */
        assert( A[iRow*N+iRow] != 0 );
        const T_PREC divisor = T_PREC(1) / A[iRow*N+iRow];
        for ( unsigned iCol = 0; iCol < N; iCol++ )
            rowSubtrahend[iCol] *= divisor;

        /* set diagonal element to 1/aii like it would be in augmented part */
        rowSubtrahend[iRow] = divisor;

        /* subtract divided iRow times multiplier from all other rows */
        for ( unsigned iRowMinuend = iRow+1; iRowMinuend < N; iRowMinuend++ )
        {
            T_PREC * const rowMinuend = &A[ iRowMinuend*N ];
            const T_PREC factor = rowMinuend[iRow];
            rowMinuend[iRow] = 0;
            for ( unsigned iCol = 0; iCol < N; ++iCol )
                rowMinuend[iCol] -= factor * rowSubtrahend[iCol];
        }
    }

    /* Now bring matrix in diagonal form
     * (everything except the upper right triangle is 0):
     *
     * iRow = 0 (begins):
     *   ( 1   b01 b02 b03 | 1/a00                   )        ^-
     *   (     1   b12 b13 | h10   1/b11             )  |*b01 |
     *   (         1   b23 | h20   h21   1/c22       )
     *   (             1   | h30   h31   h32   1/d33 )
     *
     * iRow = 1 (begins):
     *   ( 1       c02 c03 | g00 g01               )                  ^-
     *   (     1   b12 b13 | h10 1/b11             )        ^-        |
     *   (         1   b23 | h20 h21   1/c22       )  |*b12 |   |*b02 |
     *   (             1   | h30 h31   h32   1/d33 )
     *
     * iRow = 2 (begins):
     *   ( 1           d03 | k00 k01 k02         )                            ^-
     *   (     1       d13 | k10 k11 k12         )                  ^-        |
     *   (         1   b23 | h20 h21 1/c22       )        ^-        |         |
     *   (             1   | h30 h31 h32   1/d33 )  |*b23 |   |*b13 |   |*b03 |
     *
     *                    <=>
     *
     * iRow = 0 (begins):
     *   ( 1/a00 \ b01     b02      b03     )  | A01=0       ^-
     *   (        -------                   )                |
     *   ( h10     1/b11 \ b12      b13     )          |*b01 |
     *   (                -------           )
     *   ( h20     h21     1/c22 \  b23     )
     *   (                        --------  )
     *   ( h30     h31     h32      1/d33 \ )
     *
     * iRow = 1 (begins):
     *   ( g00  g01   | b02      b03     )              |A02=0 ^-
     *   (            |                  )                     |
     *   ( h10  1/b11 \ b12      b13     ) |A12=0 ^-           |
     *   (             -------           )        |            |
     *   ( h20  h21     1/c22 \  b23     )        |*b12        |*b02
     *   (                     --------  )
     *   ( h30  h31     h32      1/d33 \ )
     *
     * In order to be able to simply subtract iRow from all rows above, we
     * first need to set the diagonal element to 1 and the elements in the
     * same column as the diagonal element to 0. Then we can simply subtract
     * over the whole row using e.g. SIMD or CUDA.
     **/
    for ( unsigned iRow = 1; iRow < N; ++iRow )
    {
        T_PREC * const rowSubtrahend = &A[ iRow*N ];
        /* save diagonal element and set it to 1 */
        const T_PREC aii = rowSubtrahend[iRow];
        //rowSubtrahend[iRow] = 1;

        /* subtract iRow from all lines above */
        for ( unsigned iRowMinuend = 0; iRowMinuend < iRow; ++iRowMinuend )
        {
            T_PREC * const rowMinuend = &A[ iRowMinuend*N ];
            /* set target in same column to 0 (but save it beforehand) */
            const T_PREC factor = rowMinuend[iRow];
            rowMinuend[iRow]    = 0;

            for ( unsigned iCol = 0; iCol < N; ++iCol )
                rowMinuend[iCol] -= factor * rowSubtrahend[iCol];
        }

        /* restore diagonal element */
        rowSubtrahend[iRow] = aii;
    }
}


template void matrixInvertGaussJacobi<float>( float * const A, const unsigned N );
template void matrixInvertGaussJacobi<double>( double * const A, const unsigned N );
template void matrixInvertGaussJacobi<long double>( long double * const A, const unsigned N );


} // namespace matrix
} // namespace math
} // namespace imresh
