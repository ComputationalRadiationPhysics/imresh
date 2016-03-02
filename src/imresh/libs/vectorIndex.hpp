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


#pragma once

#include <vector>


namespace imresh
{
namespace libs
{

    /**
     * converts a vector index (i,j) to a linear index i*Nx+j
     *
     * If we have 2 slates of 3x4 length and we wanted the index  [i,j,k] =
     * [1,2,1] (begin with 0!), then the matrix lies in the memory like:
     *   [oooo|oooo|oooo] [oooo|oooo|oxoo]
     * This means to address that element (above marked x) we need to
     * calculate:
     *   lini = k + j*n_k + i*n_j*n_k
     *   21   = 1 + 2*4   + 1*3*4
     *
     * @param[in] rIndex vector index, i.e. (i) or (i,j) or (i,j,k) or
     *            (i0,i1,i2,i3,...)
     * @param[in] rnSize length of dimensions e.g. (4,5) for 2D i.e. 4 elements
     *            in the first dimension and 5 in the second, meaning always 5
     *            elements lie contiguous in memory
     * @return linear index
     **/
    unsigned int convertVectorToLinearIndex
    (
        std::vector<unsigned int> const rIndex,
        std::vector<unsigned int> const rnSize
    );

    /**
     * Reverses the effect of @see convertVectorToLinearIndex
     *
     * To reverse the equation
     *   lini = i9 + i8*n9 + i7*n9*n8 + i6*n9*n8*n7 + ... + i0*n9*...*n1
     *   21   = 1  + 2*4   + 1*3*4 + 0
     * we can use subsequent modulos
     *   k   = 21 mod (n9=4) = 1
     *   tmp = 21  /  (n9=4) = 5
     *   j   = 5  mod (n8=3) = 2
     *   tmp = 5   /  (n8=3) = 1
     *   i   = 1  mod (n7=2) = 1
     *      ...
     **/
    std::vector<unsigned int> convertLinearToVectorIndex
    (
        unsigned int const rLinIndex,
        std::vector<unsigned int> const rnSize
    );


    /**
     * shifts an n-dimensional index by half the size
     *
     * The use case is the FFT where the 0 frequency is at the 0th array entry
     * but we want it in the center, in all dimensions.
     *
     * @param[in] rLinearIndex simple linear index which should be in
     *            [0,product(rDim))
     * @param[in] rSize the size of each dimension
     **/
    unsigned fftShiftIndex
    (
        unsigned int rLinearIndex,
        std::vector<unsigned int> const rSize
    );


} // namespace libs
} // namespace imresh
