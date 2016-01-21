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

int main( void )
{
    // First step is to initialize the library.
    imresh::io::taskQueueInit( );

    // Read in a HDF5 file containing a grey scale image as a table
    auto file = imresh::io::readInFuncs::readHDF5( "../examples/imresh" );
    // This step is only needed because we have no real images
    imresh::libs::diffractionIntensity( file.first, file.second );

    // Now we can run the algorithm for testing purposes and free the data
    // afterwards
    imresh::io::addTask( file.first,
                          file.second,
                          imresh::io::writeOutFuncs::justFree,
                          "free" /* gives an identifier for debugging */ );

    // Now let's test the PNG output
#   ifdef USE_PNG
        // Let's see, how the images look after several different time steps.
        for( int i = 1; i < 10; i++)
        {
            // First read that HDF5 file once again (because the memory is
            // overwritten)
            file = imresh::io::readInFuncs::readHDF5( "../examples/imresh" );
            // Again, this step is only needed because we have no real images
            imresh::libs::diffractionIntensity( file.first, file.second );
            imresh::io::addTask( file.first,
                                  file.second,
                                  imresh::io::writeOutFuncs::writeOutPNG,
                                  "imresh_" + std::to_string( i ) + "_cycles.png",
                                  i /* sets the number of iterations */ );
        }
#   endif

    // How about the HDF5 output?
#   ifdef USE_SPLASH
        // First read that HDF5 file once again (because the memory is
        // overwritten)
        file = imresh::io::readInFuncs::readHDF5( "../examples/imresh" );
        // Again, this step is only needed because we have no real images
        imresh::libs::diffractionIntensity( file.first, file.second );
        imresh::io::addTask( file.first,
                              file.second,
                              imresh::io::writeOutFuncs::writeOutHDF5,
                              "imresh_out" );
#   endif

    // The last step is always deinitializing the library.
    imresh::io::taskQueueDeinit( );

    return 0;
}
