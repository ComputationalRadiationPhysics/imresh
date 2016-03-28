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
#include <iomanip>          // setw, setfill
#include <string>           // std::string
#include <sstream>
#include <cstring>          // memcpy

#include "io/taskQueue.hpp"
#include "io/readInFuncs/readInFuncs.hpp"
#include "io/writeOutFuncs/writeOutFuncs.hpp"
#include "libs/diffractionIntensity.hpp"
#include "libs/fftShift.hpp"
#ifndef USE_SPLASH
#   include "createTestData/createAtomCluster.hpp"
#endif


int main( void )
{
    // First step is to initialize the library.
    imresh::io::taskQueueInit( );

#   ifdef USE_SPLASH
        // Read in a HDF5 file containing a grey scale image as a table
        auto file = imresh::io::readInFuncs::readHDF5( "../examples/testData/imresh" );
#   else
        using namespace examples::createTestData;

        /* using 'file' here in order to be compliant with USE_SPLASH version */
        ImageDimensions imageSize { 300, 300 };
        std::pair<float *,ImageDimensions> file
        {
            createAtomCluster( imageSize.first, imageSize.second ),
            imageSize
        };
#   endif
    // This step is only needed because we have no real images
    imresh::libs::diffractionIntensity( file.first, file.second.first, file.second.second );
    // And now we free it again.
    if( file.first != NULL )
    {
        delete[] file.first;
        file.first = NULL;
    }

    // Now let's test the PNG in- and output
#   ifdef USE_PNG
        // Let's see, how the images look after several different time steps.
        file = imresh::io::readInFuncs::readPNG( "../examples/testData/atomCluster-extent-diffractionIntensity.png" );
            imresh::libs::fftShift( file.first, file.second.first, file.second.second );
        auto imageWidth = file.second.first;
        auto imageHeight = file.second.second;
        for( auto i = 1u; i <= 18; i++ )
        {
            // Again, this step is only needed because we have no real images
            //imresh::libs::diffractionIntensity( file.first, file.second.first, file.second.second );
            // beware diffraction intensity will be sent to the default stream thereby serializing the addTask calls!!
            auto tmpTestImage = new float[ imageWidth * imageHeight ];
            memcpy( tmpTestImage, file.first, imageWidth * imageHeight * sizeof( tmpTestImage[0] ) );

            std::ostringstream filename;
            filename << "imresh_" << std::setw( 2 ) << std::setfill( '0' )
                     << i << "_cycles.png";

            std::cout << "[threadedExample] Starting Task " << i << std::endl;
            imresh::io::addTask(
                // writeOutAndFreePNG calls delete[]. pngwriter hangs the whole thing. Writing out to PNG takes almost as long as a kernel, thereby serialising the shrink-wrap calls ...
                imresh::io::writeOutFuncs::justFree,
                filename.str(),
                tmpTestImage, imageWidth, imageHeight,
                32 /* sets the number of iterations */ );
        }
#   endif

    // How about the HDF5 output?
#   ifdef USE_SPLASH
        // First read that HDF5 file once again (because the memory is
        // overwritten) BEWARE! This path is dependent on the folder structure!
        file = imresh::io::readInFuncs::readHDF5( "../examples/testData/imresh" );
        // Again, this step is only needed because we have no real images
        imresh::libs::diffractionIntensity( file.first, file.second.first, file.second.second );
        imresh::io::addTask(
            imresh::io::writeOutFuncs::writeOutAndFreeHDF5,
            "imresh_out",
            file.first,
            file.second.first,
            file.second.second
        );
#   endif

    // The last step is always deinitializing the library.
    imresh::io::taskQueueDeinit( );

    return 0;
}
