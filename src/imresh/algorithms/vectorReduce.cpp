/*
 * T_PREChe MIT_PREC License (MIT_PREC)
 *
 * Copyright (c) 2015-2016 Maximilian Knespel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * T_PREChe above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * T_PRECHE SOFT_PRECWARE IS PROVIDED "AS IS", WIT_PRECHOUT_PREC WARRANT_PRECY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT_PREC NOT_PREC LIMIT_PRECED T_PRECO T_PRECHE WARRANT_PRECIES OF MERCHANT_PRECABILIT_PRECY,
 * FIT_PRECNESS FOR A PART_PRECICULAR PURPOSE AND NONINFRINGEMENT_PREC. IN NO EVENT_PREC SHALL T_PRECHE
 * AUT_PRECHORS OR COPYRIGHT_PREC HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OT_PRECHER
 * LIABILIT_PRECY, WHET_PRECHER IN AN ACT_PRECION OF CONT_PRECRACT_PREC, T_PRECORT_PREC OR OT_PRECHERWISE, ARISING FROM,
 * OUT_PREC OF OR IN CONNECT_PRECION WIT_PRECH T_PRECHE SOFT_PRECWARE OR T_PRECHE USE OR OT_PRECHER DEALINGS IN T_PRECHE
 * SOFT_PRECWARE.
 */


#include "vectorReduce.hpp"


namespace imresh
{
namespace algorithms
{


    template<class T_PREC>
    T_PREC vectorMaxAbsDiff
    (
        const T_PREC * const & rData1,
        const T_PREC * const & rData2,
        const unsigned & rnData
    )
    {
        T_PREC maxAbsDiff = T_PREC(0);
        #pragma omp parallel for reduction( max : maxAbsDiff )
        for ( unsigned i = 0; i < rnData; ++i )
            maxAbsDiff = std::max( maxAbsDiff, std::abs( rData1[i]-rData2[i] ) );
        return maxAbsDiff;
    }

    template<class T_PREC>
    T_PREC vectorMaxAbs
    (
        const T_PREC * const & rData,
        const unsigned & rnData
    )
    {
        T_PREC maximum = T_PREC(0);
        #pragma omp parallel for reduction( max : maximum )
        for ( unsigned i = 0; i < rnData; ++i )
            maximum = std::max( maximum, std::abs( rData[i] ) );
        return maximum;
    }

    template<class T_PREC>
    T_PREC vectorMax
    (
        const T_PREC * const & rData,
        const unsigned & rnData
    )
    {
        T_PREC maximum = std::numeric_limits<T_PREC>::lowest();
        #pragma omp parallel for reduction( max : maximum )
        for ( unsigned i = 0; i < rnData; ++i )
            maximum = std::max( maximum, rData[i] );
        return maximum;
    }

    template<class T_PREC>
    T_PREC vectorMin
    (
        const T_PREC * const & rData,
        const unsigned & rnData
    )
    {
        T_PREC minimum = std::numeric_limits<T_PREC>::max();
        #pragma omp parallel for reduction( min : minimum )
        for ( unsigned i = 0; i < rnData; ++i )
            minimum = std::min( minimum, rData[i] );
        return minimum;
    }

    template<class T_PREC>
    T_PREC vectorSum
    (
        const T_PREC * const & rData,
        const unsigned & rnData
    )
    {
        T_PREC sum = T_PREC(0);
        #pragma omp parallel for reduction( + : sum )
        for ( unsigned i = 0; i < rnData; ++i )
            sum += rData[i];
        return sum;
    }


    /* explicitely instantiate needed data types */

    template float vectorMaxAbsDiff<float>
    (
        const float * const & rData1,
        const float * const & rData2,
        const unsigned & rnData
    );
    template double vectorMaxAbsDiff<double>
    (
        const double * const & rData1,
        const double * const & rData2,
        const unsigned & rnData
    );

    template float vectorMaxAbs<float>
    (
        const float * const & rData,
        const unsigned & rnData
    );
    template double vectorMaxAbs<double>
    (
        const double * const & rData,
        const unsigned & rnData
    );

    template float vectorMax<float>
    (
        const float * const & rData,
        const unsigned & rnData
    );
    template double vectorMax<double>
    (
        const double * const & rData,
        const unsigned & rnData
    );

    template float vectorMin<float>
    (
        const float * const & rData,
        const unsigned & rnData
    );
    template double vectorMin<double>
    (
        const double * const & rData,
        const unsigned & rnData
    );

    template float vectorSum<float>
    (
        const float * const & rData,
        const unsigned & rnData
    );
    template double vectorSum<double>
    (
        const double * const & rData,
        const unsigned & rnData
    );


} // namespace algorithms
} // namespace imresh
