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


#include "gaussian.tpp"


namespace imresh
{
namespace libs
{


    /* Explicitely instantiate certain template arguments to make an object
     * file. Furthermore this saves space, as we don't need to write out the
     * data types of all functions to instantiate */

    #define __INSTANTIATE_TMP(T_PREC)   \
    template void applyKernel<T_PREC>   \
    (                                   \
        T_PREC       * const rData    , \
        unsigned int   const rnData   , \
        const T_PREC * const rWeights , \
        unsigned int   const rnWeights, \
        unsigned int   const rnThreads  \
    );
    __INSTANTIATE_TMP(float)
    __INSTANTIATE_TMP(double)
    #undef __INSTANTIATE_TMP


    #define __INSTANTIATE_TMP(T_PREC)   \
    template void gaussianBlur<T_PREC>  \
    (                                   \
        T_PREC     * const rData  ,     \
        unsigned int const rnDataX,     \
        double       const rSigma       \
    );
    __INSTANTIATE_TMP(float)
    __INSTANTIATE_TMP(double)
    #undef __INSTANTIATE_TMP


    #define __INSTANTIATE_TMP(NAME,T_PREC)      \
    template void gaussianBlur##NAME<T_PREC>    \
    (                                           \
        T_PREC     * const rData  ,             \
        unsigned int const rnDataX,             \
        unsigned int const rnDataY,             \
        double       const rSigma               \
    );
    __INSTANTIATE_TMP(,float)
    __INSTANTIATE_TMP(,double)
    __INSTANTIATE_TMP(Vertical,float)
    __INSTANTIATE_TMP(Vertical,double)
    __INSTANTIATE_TMP(VerticalUncached,float)
    __INSTANTIATE_TMP(VerticalUncached,double)
    __INSTANTIATE_TMP(Horizontal,float)
    __INSTANTIATE_TMP(Horizontal,double)
    #undef __INSTANTIATE_TMP


} // namespace libs
} // namespace imresh
