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


#include "vectorIndex.h"


namespace libs
{


    unsigned convertVectorToLinearIndex
    (
        const std::vector<unsigned> & rIndex,
        const std::vector<unsigned> & rnSize
    )
    {
        /* check sanity of input arguments */
#       ifndef NDEBUG
            assert( rIndex.size() == rnSize.size() );
            for ( unsigned i = 0; i < rIndex.size(); ++i )
            {
                assert( rIndex[i] < rnSize[i] );
            }
#       endif

        /* convert vector index, e.g. for 10 dimensions:
         *   lini = i9 + i8*n9 + i7*n9*n8 + i6*n9*n8*n7 + ... + i0*n9*...*n1 */
        unsigned linIndex  = 0;
        unsigned prevRange = 1;
        for ( int i = (int) rnSize.size() - 1; i >= 0; --i )
        {
            linIndex  += rIndex[i] * prevRange;
            prevRange *= rnSize[i];
        }
        return linIndex;
    }

    std::vector<unsigned> convertLinearToVectorIndex
    (
        unsigned rLinIndex,
        const std::vector<unsigned> & rnSize
    )
    {
        /* sanity checks for input parameters */
#       ifndef NDEBUG
            unsigned maxRange = 1;
            for ( const auto & nDimI : rnSize )
            {
                assert( nDimI > 0 );
                maxRange *= nDimI;
            }
            assert( rLinIndex < maxRange );
#       endif

        std::vector<unsigned> vecIndex( rnSize.size() );
        for ( int i = (int) rnSize.size() - 1; i >= 0; --i )
        {
            vecIndex[i]  = rLinIndex % rnSize[i];
            rLinIndex   /= rnSize[i];
        }

        assert( rLinIndex == 0 );

        return vecIndex;
    }

    
    unsigned fftShiftIndex
    (
        const unsigned & rLinearIndex,
        const std::vector<unsigned> & rSize
    )
    {
        std::vector<unsigned> vectorIndex =
            convertLinearToVectorIndex( rLinearIndex, rSize );
        for ( unsigned i = 0; i < rSize.size(); ++i )
        {
            vectorIndex[i] += rSize[i] / 2;
            vectorIndex[i] %= rSize[i];
        }
        return convertVectorToLinearIndex( vectorIndex, rSize );
    }


} // namespace libs
