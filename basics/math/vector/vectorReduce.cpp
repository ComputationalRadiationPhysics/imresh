
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

template<class T>
T vectorMaxAbs( T * const rData, const unsigned rnData )
{
    T maximum = T(0);
    #pragma omp parallel for reduction( max : maximum )
    for ( unsigned i = 0; i < rnData; ++i )
        maximum = std::max( maximum, std::abs( rData[i] ) );
    return maximum;
}

template<class T>
T vectorMax( T * const rData, const unsigned rnData )
{
    T maximum = T(0);
    #pragma omp parallel for reduction( max : maximum )
    for ( unsigned i = 0; i < rnData; ++i )
        maximum = std::max( maximum, rData[i] );
    return maximum;
}

template<class T>
T vectorMin( T * const rData, const unsigned rnData )
{
    T minimum = T(0);
    #pragma omp parallel for reduction( min : minimum )
    for ( unsigned i = 0; i < rnData; ++i )
        minimum = std::min( minimum, rData[i] );
    return minimum;
}


/* explicitely instantiate needed data types */
template<class T> void dummyFunction( void )
{
    vectorMaxAbsDiff<T>( NULL, NULL, 0 );
    vectorMaxAbs<T>( NULL, 0 );
    vectorMax<T>( NULL, 0 );
    vectorMin<T>( NULL, 0 );
}
template void dummyFunction<float>();
template void dummyFunction<double>();


} // namespace vector
} // namespace math
} // namespace imresh
