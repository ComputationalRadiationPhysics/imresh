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


#include "createAtomCluster.h"
#include "diffractionIntensity.h"
#include "shrinkWrap.h"
//#include "cudaShrinkWrap.h"

int main( void )
{
    using namespace imresh;
    using namespace examples;
    using namespace libs;

    std::vector<unsigned> imageSize = {160,160};
    float * pAtomCluster = createAtomCluster( imageSize );
    diffractionIntensity( pAtomCluster, imageSize );
    shrinkWrap( pAtomCluster, imageSize, 64 /*cycles*/, 1e-6 /* targetError */ );
    //cudaShrinkWrap( pAtomCluster, imageSize, 64 /*cycles*/, 1e-6 /* targetError */ );
    /* pAtomCluster now holds the original image again (with some deviation)
     * you could compare the current state with the data returned by
     * createAtomCluster now */
    delete[] pAtomCluster;

    return 0;
}