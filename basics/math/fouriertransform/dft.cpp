#include "dft.h"

namespace imresh {
namespace math {
namespace fouriertransform {


template<class T_PREC>
void dft
( std::complex<T_PREC> * rData, const unsigned rnData, const bool rForward )
{
    using complex = std::complex<T_PREC>;
    complex * result = new complex[rnData];

    const complex arg = complex{ /*re*/0,/*im*/ rForward ? -1.0f : 1.0f }
                      * T_PREC(2.0*M_PI / rnData);
    const T_PREC a = rForward ? 1 : T_PREC(1) / T_PREC(rnData);

    for ( unsigned k = 0; k < rnData; ++k )
    {
        result[k] = 0;
        for ( unsigned n = 0; n < rnData; ++n )
            result[k] += a * rData[n] * std::exp( arg*T_PREC(k*n) );
    }
    memcpy( rData, result, rnData*sizeof(rData[0]) );

    delete[] result;
}

template void dft<float>( std::complex<float> * rData, const unsigned rnData, const bool rForward );
template void dft<double>( std::complex<double> * rData, const unsigned rnData, const bool rForward );
template void dft<long double>( std::complex<long double> * rData, const unsigned rnData, const bool rForward );

} // namespace fouriertransform
} // namespace math
} // namespace imresh
