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

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>           // std::string
#include <cstdio>

#include "libs/diffractionIntensity.hpp"
#include "algorithms/shrinkWrap.hpp"                // fftShift, shrinkWrap
#include "algorithms/cuda/cudaShrinkWrap.h"
#include "createTestData/createAtomCluster.hpp"
#include "io/writeOutFuncs/writeOutFuncs.hpp"
#include "io/readInFuncs/readInFuncs.hpp"


int main( int argc, char ** argv )
{
    using namespace imresh::io::writeOutFuncs;
    using ImageDimensions = std::pair<unsigned int, unsigned int>;

    std::vector<unsigned> imageSize { 6000, 3000};
    float * pAtomCluster;

    if ( argc > 1 )
    {
        #if USE_PNG
            auto file = imresh::io::readInFuncs::readPNG( argv[1] );
            imageSize[0] = file.second.first;
            imageSize[1] = file.second.second;
            pAtomCluster = file.first;
            printf( "name: %s, memory: %p, width: %u, height: %u\n", argv[1], pAtomCluster, imageSize[0], imageSize[1] );
            writeOutPNG( pAtomCluster, ImageDimensions{ imageSize[0], imageSize[1] }, "atomCluster-object.png" );
        #endif
    }
    else
    {
        using namespace examples::createTestData;
        pAtomCluster = createAtomCluster( imageSize[0], imageSize[1] );
        #if USE_PNG
            writeOutPNG( pAtomCluster, ImageDimensions{ imageSize[0], imageSize[1] }, "atomCluster-object.png" );
        #endif

        imresh::libs::diffractionIntensity( pAtomCluster, imageSize[0], imageSize[1] );
        #if USE_PNG
            using imresh::algorithms::fftShift;
            fftShift( pAtomCluster, imageSize[0], imageSize[1] );
            writeOutPNG( pAtomCluster, ImageDimensions{ imageSize[0], imageSize[1] }, "atomCluster-diffractionIntensity.png" );
            fftShift( pAtomCluster, imageSize[0], imageSize[1] );
        #endif
    }

    #if USE_FFTW
        imresh::algorithms::shrinkWrap( pAtomCluster, imageSize, 64 /*cycles*/, 1e-6 /* targetError */ );
    #else
        imresh::algorithms::cuda::cudaShrinkWrap( pAtomCluster, imageSize[0], imageSize[1], cudaStream_t(0) /* stream */, 64 /*cycles*/, 1e-6 /* targetError */ );
    #endif
    /* pAtomCluster now holds the original image again (with some deviation)
     * you could compare the current state with the data returned by
     * createAtomCluster now */

    #if USE_PNG
        writeOutPNG( pAtomCluster, ImageDimensions{ imageSize[0], imageSize[1] }, "atomCluster-reconstructed.png" );
    #endif

    delete[] pAtomCluster;

    return 0;
}
