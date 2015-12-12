

namespace numeric {
namespace integrate {

void cacheIntegrationCoefficients(void)
{
    if ( not wiab.empty() )
        return;

    #include "initwiab.cpp"
}

template<typename T_PREC, typename T_VEC>
T_PREC newtonCotes( const T_VEC & f, T_PREC h, int n = 0 )
{
    cacheIntegrationCoefficients();

    const int N = (int) f.size();
    if ( n == 0)
        n = std::min( N, (int)wiab[0][0][0] ); /* wiab[0][0] is supposed to store maximum key value n */

    #ifndef NDBEUG
        if ( not (n <= N) )
            printf("n:%i, N:%i\n", n,N );
        assert( n <= N && "NewtonCotes was called with an error order n requiring more sampling points N than given" );
    #endif

    double sum = 0;
    const auto & w = wiab[n][ std::min( N, 2*n+1 ) ]; // can be ublas::vector, or Vec<double>, or std::vector
    const int Nw = (int) w.size();

    if ( N <= 2*n+1 ) {
        #ifndef NDEBUG
            bool toassert = (Nw == N);
            if (not toassert) {
                std::cout << "(Nw=" << Nw << ") == (N=" << N << ") failed, meaning the number of weightings is not equal the number of sampling points of f! This could happen if n was chosen greater than coefficients do exist!\n";
                assert( Nw == N );
            }
        #endif
        for ( int i = 0; i < N; ++i )
            sum += w[i]*f[i];
    }
    else
    {
        assert( Nw < N );
        int i = -1;
        do {
            ++i;
            #ifndef NDEBUG
                assert( i < N && "Weightings for integration incorrect, as no coefficient being 1 could be found, even though N>2n. Maybe you forgot to call cacheIntegrationCoefficients()" );
                if ( not (i < Nw) ) {
                    printf( "N:%i, n:%i, min(N,2n+1)=%i, ...\n", N,n, std::min( N, 2*n+1 ) );
                    assert( i < Nw && "Couldn't find weighting coefficient being 1.0. Internal error! Check initwiab.cpp for correctness" );
                }
            #endif
            sum += w[i]*f[i];
        } while( w[i] != 1. ); // floating point exact comparison wanted! Need to set the last value in weightings to 1.
        const int imiddle = i;
        const int Nright  = Nw-(imiddle+1); /* e.g. Nw=5, i=0,1,2,3,4; imiddle=2 => Nright = 5-(2+1) = 2 */

        for (i = imiddle+1; i < N-Nright; ++i )
        {
            assert( i < N );
            sum += f[i];
        }

        /* do N-Nright, N-Nright+1, ..., N-1. E.g. N=7, Nright 2 => do 5,6 */
        assert( i == N-Nright );
        int j;
        for (j = imiddle+1; j < Nw; ++j )
        {
            assert( j < Nw && i < N );
            sum += w[j]*f[i];
            i++;
        }
        /* test if we really did use every coefficient and function value */
        assert( j == Nw );
        assert( i == N  );
    }

    return sum*h;
}

template<typename T_PREC, typename T_VEC>
inline T_PREC trapezoid (const T_VEC & f, T_PREC h)
{
    return newtonCotes(f,h,2);
}

template<typename T_PREC, typename T_VEC>
inline T_PREC simpson (const T_VEC & f, T_PREC h)
{
    return newtonCotes(f,h,3);
}


} // namespace integrate

} // namespace numeric

