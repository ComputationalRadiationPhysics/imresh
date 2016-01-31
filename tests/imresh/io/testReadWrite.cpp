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


#include "testReadWrite.hpp"


#include <cstdlib>          // srand, rand
#include <cstring>          // memcpy
#include <cfloat>           // FLT_MAX
#include <cmath>            // fmin
#include <chrono>           // high_resolution_clock
#include <iostream>
#include <iomanip>          // setw
#include <cassert>
#include "io/readInFuncs/readInFuncs.hpp"
#include "io/writeOutFuncs/writeOutFuncs.hpp"
#include "benchmarkHelper.hpp"  // getLogSpacedSamplingPoints


namespace imresh
{
namespace io
{


    /* in order to filter out page time outs or similarly long random wait
     * times, we repeat the measurement nRepetitions times and choose the
     * shortest duration measured */
    unsigned int constexpr nRepetitions = 10;
    unsigned int constexpr nMaxElements = 1024*1024;  // ~8000 x 8000 px
    const char * tmpFileName = "blfh8921721afh";


#ifdef USE_PNG

    void testPng( void )
    {
        using namespace std::chrono;
        using namespace imresh::io::readInFuncs;
        using namespace imresh::io::writeOutFuncs;
        using imresh::tests::getLogSpacedSamplingPoints;
        using ImageDim = std::pair<unsigned int, unsigned int>;

        /**
         * The test consists of writing out some data and reading it again
         * and check if the data is equal.
         * By using test data, which was alread written and read through
         * pngwriter we can ignore floating point rounding errors.
         * If the data is not exactly equal, then this means, then multiple
         * read/writes result in non-convergent data degradation!
         **/

        decltype( readPNG( "" ) ) file;
        auto tmpSaved  = new float[ nMaxElements ];
        auto tmpSaved0 = new float[ nMaxElements ];

        using clock = std::chrono::high_resolution_clock;
        decltype( clock::now() ) clock0, clock1;
        duration<double> seconds;

        std::cout << "\n";
        std::cout << "Timings in milliseconds:\n";
        std::cout << "image size (nCols,nRows) : memcpy | writeOutPNG | readInPNG |\n";

        for ( auto nElements : getLogSpacedSamplingPoints( 2, nMaxElements, 20 ) )
        {
            unsigned int const Nx  = floor(sqrt( nElements ));
            unsigned int const Ny = Nx;
            assert( Nx*Ny<= nMaxElements );
            nElements = Nx * Ny;
            /* this case is not supported by readPNG, because pngwriter lacks
             * proper error handling
             * @see https://github.com/pngwriter/pngwriter/issues/82 */
            if ( Nx == 1 and Ny == 1 )
                continue;

            {
                auto pData = new float[ nElements ];
                srand( 876512789 );
                for ( auto i = 0; i < nElements; ++i )
                    pData[i] = (float) rand() / RAND_MAX;

                /* note that this call deletes the pointer @todo: move delete[] inside taskQueue.cu */
                writeOutPNG( pData, ImageDim{ Nx, Ny }, tmpFileName );

                file = readPNG( tmpFileName );
                assert( file.second.first == Nx );
                assert( file.second.first == Ny );
            }
            memcpy( tmpSaved0, file.first, nElements * sizeof( file.first[0] ) );

            std::cout << "           ";
            std::cout << "(" << std::setw(5) << Nx << ","
                             << std::setw(5) << Ny << ") : ";

            /* memcpy */
            float minTime = FLT_MAX;
            for ( auto iRepetition = 0u; iRepetition < nRepetitions;
                  ++iRepetition )
            {
                clock0 = clock::now();
                memcpy( tmpSaved, tmpSaved0,
                        Nx * Ny * sizeof( file.first[0] ) );
                clock1 = clock::now();
                seconds = duration_cast<duration<double>>( clock1 - clock0 );
                minTime = fmin( minTime, seconds.count() * 1000 );
            }
            std::cout << std::setw(8) << minTime << " | " << std::flush;

            float minTimeRead  = FLT_MAX;
            float minTimeWrite = FLT_MAX;
            for ( auto iRepetition = 0u; iRepetition < nRepetitions;
                  ++iRepetition )
            {
                /* write */
                clock0 = clock::now();
                    writeOutPNG( file.first, ImageDim{ Nx, Ny }, tmpFileName );
                clock1 = clock::now();
                seconds = duration_cast<duration<double>>( clock1 - clock0 );
                minTimeRead = fmin( minTimeRead, seconds.count() * 1000 );

                /* read */
                clock0 = clock::now();
                    file = readPNG( tmpFileName );
                clock1 = clock::now();
                seconds = duration_cast<duration<double>>( clock1 - clock0 );
                minTimeWrite = fmin( minTimeWrite, seconds.count() * 1000 );

                for ( auto i = 0u; i < Nx*Ny; ++i )
                {
                    if( not ( file.first[i] == tmpSaved0[i] ) )
                    {
                        printf( "read from png pixel %i: %f != %f read initially\n", i, file.first[i], tmpSaved[i] );
                        assert( file.first[i] == tmpSaved0[i] );
                    }
                }
            }
            std::cout << std::setw(8) << minTimeRead << " | " << std::flush;
            std::cout << std::setw(8) << minTimeWrite << " | \n" << std::flush;
        }

        delete[] file.first;
    }

#endif


} // namespace algorithms
} // namespace imresh
