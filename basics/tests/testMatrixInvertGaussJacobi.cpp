
#include <cassert>
#include <cstdlib>  // RAND_MAX, rand, srand
#include <cfloat>   // FLT_EPSILON
#include <iostream>

#include "math/matrix/matrixInvertGaussJacobi.h"


namespace imresh {
namespace test {


float testInverseUnity( const float * const A, const float * const B, const int N )
{
    /* matrix multiply to test if result is unit matrix */
    float absErr = 0;
    for ( int iCol = 0; iCol < N; ++iCol )
        for ( int iRow = 0; iRow < N; ++iRow )
        {
            float sum = 0;
            for ( int k = 0; k < N; k++ )  // dot product
               sum += A[iRow*N+k] * B[k*N+iCol];
            absErr += std::abs( ( iRow == iCol ? 1 : 0 ) - sum );
        }
    return absErr;
}

void testMatrixInvertGaussJacobi(void)
{
    using namespace imresh::math::matrix;

    srand(1468467);
    const int nTestRuns = 10;
    const int Nmax = (nTestRuns+1)*2;
    float * A = new float[Nmax*Nmax];
    float * B = new float[Nmax*Nmax];

    /* @todo: it should be very very very rare, but it could happen, that
     * the rank of the random matrix is actually smaller than N, meaning the
     * matrix inversion will fail. I don't check for that yet! */
    for ( int N = 1; N < Nmax; ++N )
    for ( int run = 0; run < nTestRuns; ++run )
    {
        for ( int i = 0; i < N*N; ++i )
            A[i] = rand() / (float) RAND_MAX;
        memcpy( B, A, sizeof(A[0])*N*N );

        matrixInvertGaussJacobi( A, N );
        float absErr = testInverseUnity(A,B,N);

        /* because we sum up all errors, the maximum error can be N*N times
         * epsilon. Furthermore the inversion calculates 2*N subtractions
         * per element. The matrix multiplication again makes N additions
         * This means the total epsilon can be 2*N*N*N*N large in the worst
         * case. I can't explain the additionally needed factor 2 to get to
         * 4*N**4, but this is only a rough estimate, so there may be other
         * errors involved */
        //std::cout << "abserr = " << absErr << " <? " << 1e-4*pow(N,4) << "\n";
        assert( absErr <= 1e-4*pow(N,4) );
    }

    /* test inversion with diagonal matrix. Roudning errors should be
     * almost non-existent in this case */
    //std::cout << "\nInvert diagonal Matrices:\n";
    for ( int N = 1; N < Nmax; ++N )
    for ( int run = 0; run < nTestRuns; ++run )
    {
        memset( A,0, N*N*sizeof(A[0]) );
        for ( int i = 0; i < N; ++i )
            A[i*N+i] = rand() / (float) RAND_MAX;
        memcpy( B, A, sizeof(A[0])*N*N );

        matrixInvertGaussJacobi( A, N );
        float absErr = testInverseUnity(A,B,N);

        //std::cout << "abserr = " << absErr << " <=? " << FLT_EPSILON*N*N << "\n";
        assert( absErr <= FLT_EPSILON*N*N );
    }

    delete[] B;
    delete[] A;
}


} // namespace imresh
} // namespace test
