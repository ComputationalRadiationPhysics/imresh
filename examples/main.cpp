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

#include <iostream>         // std::cout, std::endl
#include <string>           // std::string
#include <utility>          // std::pair

#include "io/taskQueue.cu"
#include "io/readInFuncs/readInFuncs.hpp"
#include "io/writeOutFuncs/writeOutFuncs.hpp"

#include "createAtomCluster.hpp"

int main( void )
{
    //auto file = imresh::io::readInFuncs::readTxt( "../PS_simple.txt" );
    std::pair<unsigned int,unsigned int> size { 100, 100 };
    float* cluster = examples::createAtomCluster( size );
    std::pair<float*,std::pair<unsigned int,unsigned int>> file { cluster, size };

    imresh::io::taskQueueInit( );

    imresh::io::addTask( file.first, file.second, imresh::io::writeOutFuncs::justFree, "free" );

#   ifdef USE_PNG
        for( int i = 1; i < 30; i++)
        {
            imresh::io::addTask( file.first, file.second, imresh::io::writeOutFuncs::writeOutPNG, "imresh_" + std::to_string( i ) + "_cycles.png", i );
        }
#   endif

#   ifdef USE_SPLASH
        imresh::io::addTask( file.first, file.second, imresh::io::writeOutFuncs::writeOutHDF5, "imresh" );
#   endif


    imresh::io::taskQueueDeinit( );

    return 0;
}
