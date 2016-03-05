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
#include "imresh/algorithms/testGaussian.hpp"
#include "imresh/algorithms/testVectorReduce.hpp"
#include "imresh/algorithms/testVectorElementwise.hpp"
#include "imresh/libs/testVectorIndex.hpp"
#include "imresh/io/testReadWrite.hpp"
//#include "imresh/algorithms/testShrinkWrap.hpp"


int main( void )
{
    using namespace imresh::algorithms;
    using namespace imresh::libs;
    using namespace imresh::io;

    TestGaussian testGaussian; testGaussian();
    testFft();
    testShrinkWrap();
    #ifdef USE_PNG
        testPng();
    #endif

    testVectorIndex();
    testUnpackBitMask();
    testCalculateHioError();
    testVectorReduce();
    testVectorElementwise();

    std::cout << "All tests OK.\n";

    return 0;
}

