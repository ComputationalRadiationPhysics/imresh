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


#include "testVectorIndex.hpp"

#include <iostream>
#include <cassert>
#include <vector>
#include <map>
#include <utility>   // pair
#include "libs/vectorIndex.hpp"


namespace imresh
{
namespace libs
{

    std::ostream & operator<<
    (
        std::ostream & rOut,
        const std::vector<unsigned> rVectorToPrint
    )
    {
        rOut << "{";
        for ( const auto & elem : rVectorToPrint )
            rOut << elem << " ";
        rOut << "}";
        return rOut;
    }

    void testVectorIndex( void )
    {
        using namespace imresh::libs;
        using Vec = std::vector<unsigned>;

        /* 1-D tests */
        const unsigned iMax = 10;
        for ( unsigned i = 0; i < iMax; ++i )
        {
            Vec vecIndex = {i};
            unsigned lini = convertVectorToLinearIndex( vecIndex, Vec{iMax} );
            assert( lini == convertVectorToLinearIndex( vecIndex, Vec{iMax+5} ) );
            assert( vecIndex == convertLinearToVectorIndex( lini, Vec{iMax} ) );
            assert( vecIndex == convertLinearToVectorIndex( lini, Vec{iMax+5} ) );
        }

        /* N-D tests */
        std::vector< std::pair< unsigned, std::pair<Vec,Vec> > > testValues =
        {
            /* linear index, {dimension size, vector index} */
            { 0, { {1,1},{0,0} } },
            { 0, { {1,3},{0,0} } },
            { 0, { {2,3},{0,0} } },

            { 0, { {1,1,1},{0,0,0} } },
            { 0, { {1,3,1},{0,0,0} } },
            { 0, { {1,3,6},{0,0,0} } },
            { 0, { {2,3,1},{0,0,0} } },

            { 2, { {1,3,1},{0,2,0} } },
            { 2, { {1,3,6},{0,0,2} } },
            { 2, { {2,3,1},{0,2,0} } },

            { 3, { {1,3,6},{0,0,3} } },
            { 3, { {2,3,1},{1,0,0} } },

            { 5, { {1,3,6},{0,0,5} } },
            { 5, { {2,3,1},{1,2,0} } },

            { 8, { {1,3,6},{0,1,2} } },

            { 2, { {1,3},{0,2} } },

            { 2, { {2,3},{0,2} } },
            { 3, { {2,3},{1,0} } },
            { 4, { {2,3},{1,1} } },
            { 5, { {2,3},{1,2} } },

            { 0, { {5,3},{0,0} } },
            { 1, { {5,3},{0,1} } },
            { 2, { {5,3},{0,2} } },
            { 3, { {5,3},{1,0} } },
            { 4, { {5,3},{1,1} } },
            { 5, { {5,3},{1,2} } },
            { 6, { {5,3},{2,0} } },
            { 7, { {5,3},{2,1} } },
            { 8, { {5,3},{2,2} } }
        };
        for ( auto const & value : testValues )
        {
            /*
            std::cout << "test lini=" << value.first
                << " =? " << convertVectorToLinearIndex( value.second.second,
                                                value.second.first )
                << " (vecIndex = " << value.second.second
                << ", dimSize = " << value.second.first << ")"
                << "\n" << std::flush;
            */
            assert( value.first ==
                    convertVectorToLinearIndex( value.second.second,
                                                value.second.first ) );
            assert( value.second.second ==
                    convertLinearToVectorIndex( value.first,
                                                value.second.first ) );
        }
    }


} // namespace tests
} // namespace imresh
