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

#include <string>
#include <sstream>
#include <vector>
#include <iostream>

#include "io/taskQueue.cu"
#include "io/readInFuncs/readInFuncs.hpp"
#include "io/writeOutFuncs/writeOutFuncs.hpp"
#include "libs/diffractionIntensity.hpp"
#include "createTestData/createAtomCluster.hpp"
#include "createTestData/createRectangle.hpp"
#include "createTestData/createCheckerboard.hpp"
#include "createTestData/createCircularSection.hpp"


namespace examples
{

    void saveToPng
    (
        float const * const data,
        std::pair< unsigned, unsigned > const & size,
        std::string const & filename
    )
    {
#       ifdef USE_PNG
            //std::cout << "size = (" << size.first << "," << size.second << ")\n";
            imresh::io::writeOutFuncs::writeOutPNG( data, size, filename );
#       endif
        delete[] data;
    }


} // namespace examples


int main( void )
{
    using namespace examples::createTestData;
    using namespace examples;

#   ifndef USE_PNG
        std::cout << "[Warning] cmake: USE_PNG=OFF, no png will be written out!\n";
#   endif

    std::vector< std::pair<unsigned,unsigned> > imageSizes {
        { 300, 200 }, { 1200, 1000 }, { 1500, 3000 } };

    for ( auto size : imageSizes )
    {
        std::stringstream filename;
        float * data;
        auto Nx = size.first;
        auto Ny = size.second;

        data = createRectangle( Nx,Ny, 0.1,0.3, 0.5,0.5, 0 );
        filename << "rectangle_" << Nx << "x" << Ny << "_0.1x0.3_0-deg.png";
        saveToPng( data, size, filename.str() ); filename.str("");

        data = createRectangle( Nx,Ny, 0.1,0.3, 0.5,0.5, 10 );
        filename << "rectangle_" << Nx << "x" << Ny << "_0.1x0.3_10-deg.png";
        saveToPng( data, size, filename.str() ); filename.str("");

        data = createRectangle( Nx,Ny, 0.1,0.05, 0.2,0.6 );
        filename << "rectangle_" << Nx << "x" << Ny << "_0.1x0.05_+0.2,0.6_10-deg.png";
        saveToPng( data, size, filename.str() ); filename.str("");

        data = createAtomCluster( Nx,Ny );
        filename << "atomCluster_" << Nx << "x" << Ny << ".png";
        saveToPng( data, size, filename.str() ); filename.str("");

        data = createCheckerboard( Nx,Ny, 0.1,0.3, 0 );
        filename << "checkerboard_" << Nx << "x" << Ny << "_0.1x0.3_0-deg.png";
        saveToPng( data, size, filename.str() ); filename.str("");

        data = createCheckerboard( Nx,Ny, 0.1,0.3, 30 );
        filename << "checkerboard_" << Nx << "x" << Ny << "_0.1x0.3_30-deg.png";
        saveToPng( data, size, filename.str() ); filename.str("");

        data = createCheckerboard( Nx,Ny, 0.05,0.05, 45 );
        filename << "checkerboard_" << Nx << "x" << Ny << "_0.05x0.05_45-deg.png";
        saveToPng( data, size, filename.str() ); filename.str("");

        data = createCircularSection( Nx,Ny, 0.2 );
        filename << "circularSection_" << Nx << "x" << Ny << "_r-0.2_+0,0_0-to-2pi.png";
        saveToPng( data, size, filename.str() ); filename.str("");

        data = createCircularSection( Nx,Ny, 0.2, 0.4,0.6, 10.*M_PI/180., 240.*M_PI/180. );
        filename << "circularSection_" << Nx << "x" << Ny << "_r-0.2_+0.4,0.6_10deg-to-170deg.png";
        saveToPng( data, size, filename.str() ); filename.str("");
    }

    return 0;
}
