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
#include "algorithms/shrinkWrap.hpp"                // shrinkWrap
#include "libs/fftShift.hpp"
#include "algorithms/cuda/cudaShrinkWrap.hpp"
#include "createTestData/createAtomCluster.hpp"
#include "io/writeOutFuncs/writeOutFuncs.hpp"
#include "io/readInFuncs/readInFuncs.hpp"


int main( int argc, char ** argv )
{
    using namespace imresh::io::writeOutFuncs;

    std::vector<unsigned> imageSize{ 600, 300 };
    float * pData;

    if ( argc > 1 )
    {
        #if USE_PNG
            std::string filename( argv[1] );
            auto file = imresh::io::readInFuncs::readPNG( filename.c_str() );
            imageSize[0] = file.second.first;
            imageSize[1] = file.second.second;
            pData = file.first;
            printf( "name: %s, memory: %p, width: %u, height: %u\n", filename.c_str(), pData, imageSize[0], imageSize[1] );
            using imresh::libs::fftShift;
            if ( argc > 2 and argv[2][0] == 'O' )
            {
                imresh::libs::diffractionIntensity( pData, imageSize[0], imageSize[1] );
                fftShift( pData, imageSize[0], imageSize[1] );
            }
            std::cout << "Write out the following values: ";
            for ( int i = 0; i < 10; ++i )
                std::cout << 65535*pData[i] << " ";
            std::cout << std::endl;
            writeOutPNG( pData, imageSize[0], imageSize[1],
                std::string( argv[1] ) + std::string( "-diffraction.png" ) );
            fftShift( pData, imageSize[0], imageSize[1] );
        #else
            std::cout << "This program was compiled without the cmake option USE_PNG, therefore it may not be given any arguments! (Normally the first argument is a path to a PNG diffraction intensity to reconstruct)" << std::endl;
            exit(1);
        #endif
    }
    else
    {
        using namespace examples::createTestData;
        pData = createAtomCluster( imageSize[0], imageSize[1] );
        #if USE_PNG
            writeOutPNG( pData, imageSize[0], imageSize[1], "atomCluster-object.png" );
        #endif

        imresh::libs::diffractionIntensity( pData, imageSize[0], imageSize[1] );
        #if USE_PNG
            using imresh::libs::fftShift;
            fftShift( pData, imageSize[0], imageSize[1] );
            writeOutPNG( pData, imageSize[0], imageSize[1], "atomCluster-diffractionIntensity.png" );
            fftShift( pData, imageSize[0], imageSize[1] );
        #endif
    }

    /* unfortunately there is a difference in the interface between
     * shrinkWrap and cudaShrinkWrap because of cuda streams */
    #if USE_FFTW
        imresh::algorithms::shrinkWrap(
            pData,
            imageSize[0],
            imageSize[1],
            64      /*cycles*/,
            1e-6    /* targetError */
        );
    #else
        imresh::algorithms::cuda::cudaShrinkWrap(
            pData,
            imageSize[0],
            imageSize[1],
            cudaStream_t(0),
            96      /* nBlocks          */,
            256     /* nThreads         */,
            64      /* cycles           */,
            1e-6    /* targetError      */,
            0       /* HioBeta (auto)   */,
            0.001   /* rIntensityCutOffAutoCorel (auto) */,
            0.01    /* rIntensityCutOff */,
            0       /* rSigma0          */,
            0       /* rSigmaChange     */
        );
    #endif
    /* pData now holds the original image again (with some deviation)
     * you could compare the current state with the data returned by
     * createAtomCluster now */

    std::string fileName;
    if ( argc > 1 )
        fileName = std::string( argv[1] ) + std::string( "-reconstructed.png" );
    else
        fileName = std::string( "atomCluster-reconstructed.png" );
    #if USE_PNG
        writeOutAndFreePNG( pData, imageSize[0], imageSize[1], fileName.c_str() );
    #else
        delete[] pData;
        std::cout << "[note] can't output the result if not compiled with the CMake-option 'USE_PNG'\n";
    #endif

    return 0;
}
