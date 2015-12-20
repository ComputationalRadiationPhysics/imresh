
#include "vectorReduce.h"


namespace imresh {
namespace math {
namespace vector {


template<class T>
T vectorMaxAbsDiff( T * const rData1, T * const rData2, const unsigned rnData )
{
    T maxAbsDiff = T(0);
    #pragma omp parallel for reduction( max : maxAbsDiff )
    for ( unsigned i = 0; i < rnData; ++i )
        maxAbsDiff = std::max( maxAbsDiff, std::abs( rData1[i]-rData2[i] ) );
    return maxAbsDiff;
}


/* explicitely instantiate needed data types */
template float vectorMaxAbsDiff<float>( float * const rData1, float * const rData2, const unsigned rnData );
template double vectorMaxAbsDiff<double>( double * const rData1, double * const rData2, const unsigned rnData );


} // namespace vector
} // namespace math
} // namespace imresh
