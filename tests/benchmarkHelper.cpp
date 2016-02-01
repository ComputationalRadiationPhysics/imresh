/*
 * The MIT License (MIT)
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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>


namespace imresh
{
namespace tests
{


    template<class T_PREC>
    T_PREC mean( std::vector<T_PREC> const vec )
    {
        auto sum = T_PREC(0);
        for ( auto const & elem : vec )
            sum += elem;
        return sum / vec.size();
    }
    template float mean<float>( std::vector<float> const vec );

    /**
     * < (x - <x>)^2 > = < x^2 + <x>^2 - 2x<x> > = <x^2> - <x>^2
     **/
    template<class T_PREC>
    T_PREC stddev( std::vector<T_PREC> const vec )
    {
        auto sum2 = T_PREC(0);
        for ( auto const elem : vec )
            sum2 += elem*elem;
        auto avg = mean( vec );
        auto const N = T_PREC( vec.size() );
        return sqrt( ( sum2/N - avg*avg )*N/(N-1) );
    }
    template float stddev( std::vector<float> const vec );


    std::vector<int> getLogSpacedSamplingPoints
    (
        const unsigned riStartPoint,
        const unsigned riEndPoint,
        const unsigned rnPoints
    )
    {
        assert( riStartPoint > 0 );
        assert( riEndPoint > riStartPoint );
        assert( rnPoints > 0 );
        assert( rnPoints <= riEndPoint-riStartPoint+1 );

        std::vector<float> naivePoints( rnPoints );
        std::vector<int> tmpPoints( rnPoints );
        std::vector<int> points( rnPoints );

        /* Naively create logspaced points rounding float to int */
        const float dx = ( logf(riEndPoint) - logf(riStartPoint) ) / rnPoints;
        const float factor = exp(dx);
        //std::cout << "dx = " << dx << ", factor = " << factor << "\n";

        naivePoints[0] = riStartPoint;
        for ( unsigned i = 1; i < rnPoints; ++i )
            naivePoints[i] = naivePoints[i-1]*factor;
        /* set last point manually because of rounding errors */
        naivePoints[ naivePoints.size()-1 ] = riEndPoint;

        //std::cout << "naivePoints = ";
        //for ( const auto & elem : naivePoints )
        //    std::cout << elem << " ";
        //std::cout << "\n";

        /* sift out values which appear twice */
        tmpPoints[0] = (int) naivePoints[0];
        unsigned iTarget = 1;
        for ( unsigned i = 1; i < rnPoints; ++i )
        {
            assert( iTarget >= 1 && iTarget < tmpPoints.size() );
            if ( tmpPoints[ iTarget-1 ] != (int) naivePoints[i] )
                tmpPoints[ iTarget++ ] = (int) naivePoints[i];
        }

        //std::cout << "tmpPoints = ";
        //for ( const auto & elem : tmpPoints )
        //    std::cout << elem << " ";
        //std::cout << "\n";

        /* if converted to int and fill in as many values as deleted */
        int nToInsert = rnPoints - iTarget;
        points[0] = tmpPoints[0];
        iTarget = 1;
        //std::cout << "nToInsert = " << nToInsert << "\n";
        for ( unsigned iSrc = 1; iSrc < rnPoints; ++iSrc )
        {
            for ( ; nToInsert > 0 and points[iTarget-1] < tmpPoints[iSrc]-1;
                  ++iTarget, --nToInsert )
            {
                assert( iTarget >= 1 && iTarget < points.size() );
                points[iTarget] = points[iTarget-1] + 1;
            }
            assert( iSrc    < tmpPoints.size() );
            //assert( iTarget < points.size() );
            /* this condition being false should only happen very rarely, if
             * rnPoints == riEndPoint - riStartPoint + 1 */
            if ( iTarget < points.size() )
                points[iTarget++] = tmpPoints[iSrc];
        }

        //std::cout << "points = ";
        //for ( const auto & elem : points )
        //    std::cout << elem << " ";
        //std::cout << "\n";

        for ( unsigned i = 1; i < points.size(); ++i )
        {
            assert( points[i-1] < points[i] );
        }

        return points;
    }


} // namespace tests
} // namespace imresh
