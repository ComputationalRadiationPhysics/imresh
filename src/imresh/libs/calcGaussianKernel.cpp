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


#include "calcGaussianKernel.hpp"
#include "calcGaussianKernel.tpp"


namespace imresh
{
namespace libs
{


    /* @todo This could almost be automatically generated from the header files ... */

    #define __INSTANTIATE( T_Prec )             \
                                                \
    template int calcGaussianKernel<T_Prec>     \
    (                                           \
        double       const rSigma   ,           \
        T_Prec *     const rWeights ,           \
        unsigned int const rnWeights,           \
        double       const rMinAbsoluteError    \
    );                                          \
                                                \
    template void calcGaussianKernel2d<T_Prec>  \
    (                                           \
        double       const rSigma    ,          \
        unsigned int const rCenterX  ,          \
        unsigned int const rCenterY  ,          \
        T_Prec *     const rWeights  ,          \
        unsigned int const rnWeightsX,          \
        unsigned int const rnWeightsY           \
    );

    __INSTANTIATE( float  )
    __INSTANTIATE( double )

    #undef __INSTANTIATE


} // namespace libs
} // namespace imresh
