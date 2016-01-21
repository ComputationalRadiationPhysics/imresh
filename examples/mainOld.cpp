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

#include <string>           // std::string

#include "io/taskQueue.cu"
#include "io/readInFuncs/readInFuncs.hpp"
#include "io/writeOutFuncs/writeOutFuncs.hpp"
#include "libs/diffractionIntensity.hpp"
#include "algorithms/shrinkWrap.hpp"
#include "algorithms/cuda/cudaShrinkWrap.h"
#include "io/taskQueue.hpp"
#include <fstream>
#include <iomanip>


void writeOut( int* mem, int size )
{
    //std::cout << &mem << std::endl;
}

int main( void )
{
    int* h_mem;
    h_mem = (int*) malloc(sizeof(int) * 100);
    auto tq = new imresh::io::taskQueue( );
    tq->addTask(h_mem, sizeof(int) * 100, (std::function<void(int*,int)>) writeOut);
    std::vector<unsigned> imageSize { 300, 300 };
    float * pAtomCluster = examples::createAtomCluster( imageSize );

    /* debug output of image */
    std::ofstream file;
    file.open("atomClusterInput.dat");
    for ( unsigned ix = 0; ix < imageSize[0]; ++ix )
    {
        for ( unsigned iy = 0; iy < imageSize[1]; ++iy )
            file << std::setw(10) << pAtomCluster[ iy*imageSize[0] + ix ] << " ";
        file << "\n";
    }
    file.close();
    std::cout << "Wrote atomClusterInput to atomClusterInput.dat\n";

    imresh::libs::diffractionIntensity( pAtomCluster, imageSize );

    file.open("diffractionIntensity.dat");
    for ( unsigned ix = 0; ix < imageSize[0]; ++ix )
    {
        for ( unsigned iy = 0; iy < imageSize[1]; ++iy )
            file << std::setw(10) << pAtomCluster[ iy*imageSize[0] + ix ] << " ";
        file << "\n";
    }
    file.close();
    std::cout << "Wrote diffractionIntensity to diffractionIntensity.dat\n";

    imresh::algorithms::shrinkWrap( pAtomCluster, imageSize, 64 /*cycles*/, 1e-6 /* targetError */ );
    //imresh::algorithms::cuda::cudaShrinkWrap( pAtomCluster, imageSize, 64 /*cycles*/, 1e-6 /* targetError */ );
    /* pAtomCluster now holds the original image again (with some deviation)
     * you could compare the current state with the data returned by
     * createAtomCluster now */

    file.open("atomClusterOutput.dat");
    for ( unsigned ix = 0; ix < imageSize[0]; ++ix )
    {
        for ( unsigned iy = 0; iy < imageSize[1]; ++iy )
            file << std::setw(10) << pAtomCluster[ iy*imageSize[0] + ix ] << " ";
        file << "\n";
    }
    file.close();
    std::cout << "Wrote atomClusterOutput to atomClusterOutput.dat\n";


    delete[] pAtomCluster;

    return 0;
}
