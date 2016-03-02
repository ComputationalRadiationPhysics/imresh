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


#include "cudaGaussian.hpp"
#include "cudaGaussian.tpp"

#include <cuda_to_cupla.hpp>    // cudaStream_t


namespace imresh
{
namespace algorithms
{
namespace cuda
{


    /* Explicitely instantiate certain template arguments to make an object file */

    #define INSTANTIATE_TMP( NAME, T_PREC )         \
    template void cudaGaussianBlur##NAME< T_PREC >  \
    (                                               \
        T_PREC * rData,                             \
        unsigned int rnDataX,                       \
        unsigned int rnDataY,                       \
        double rSigma,                              \
        cudaStream_t rStream,                       \
        bool rAsync                                 \
    );

    INSTANTIATE_TMP( , float )
    INSTANTIATE_TMP( , double )
    INSTANTIATE_TMP( SharedWeights, float )
    INSTANTIATE_TMP( SharedWeights, double )
    INSTANTIATE_TMP( Vertical, float )
    INSTANTIATE_TMP( Vertical, double )
    INSTANTIATE_TMP( HorizontalSharedWeights, float )
    INSTANTIATE_TMP( HorizontalSharedWeights, double )

    #undef INSTANTIATE_TMP


} // namespace cuda
} // namespace algorithms
} // namespace imresh
