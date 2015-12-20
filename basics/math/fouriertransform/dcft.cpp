#include "dcft.h"


namespace imresh {
namespace math {
namespace fouriertransform {


template<class F, class G>
float trigScp(const F & f, const G & g)
{
    const float a  = -M_PI;
    const float b  =  M_PI;
    const float N  = 1e5;
    const float dx = (b-a) / N;

    struct Integrand {
        const F & f; const G & g;
        int N; float a; float dx;
        int size(void) const { return N; }
        float operator[]( int i ) const { return f(a+i*dx)*g(a+i*dx); }
    } integrand({ f,g,(int)N,a,dx });

    return numeric::integrate::trapezoid( integrand, dx ) / M_PI;
}

template<class F>
void dcft( F f, int rnCoefficients, float * rCoefficients )
{
    for ( int i = 0; i < rnCoefficients; ++i )
        rCoefficients[i] = trigScp( f, CosBase({i}) ) / M_PI;
    for ( int i = 0; i < rnCoefficients; ++i )
        rCoefficients[rnCoefficients+i] = trigScp( f, SinBase({i}) ) / M_PI;
}


} // namespace fouriertransform
} // namespace math
} // namespace imresh
