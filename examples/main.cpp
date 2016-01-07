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

#include <functional>

#include "createAtomCluster.hpp"
#include "libs/diffractionIntensity.hpp"
//#include "algorithms/shrinkWrap.hpp"
#include "algorithms/cuda/cudaShrinkWrap.h"
#include "io/taskQueue.hpp"

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
    std::vector<unsigned> imageSize { 160, 160 };
    float * pAtomCluster = examples::createAtomCluster( imageSize );
    imresh::libs::diffractionIntensity( pAtomCluster, imageSize );
    //imresh::algorithms::shrinkWrap( pAtomCluster, imageSize, 64 /*cycles*/, 1e-6 /* targetError */ );
    imresh::algorithms::cuda::cudaShrinkWrap( pAtomCluster, imageSize, 64 /*cycles*/, 1e-6 /* targetError */ );
    /* pAtomCluster now holds the original image again (with some deviation)
     * you could compare the current state with the data returned by
     * createAtomCluster now */
    delete[] pAtomCluster;

    return 0;
}
