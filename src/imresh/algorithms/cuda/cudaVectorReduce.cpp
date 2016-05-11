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


#include "cudaVectorReduce.tpp"


namespace imresh
{
namespace algorithms
{
namespace cuda
{


    /* explicit template instantiations */

    #define __INSTANTIATE_TMP(NAME,T_PREC)  \
    template                                \
    T_PREC cudaVector##NAME<T_PREC>         \
    (                                       \
        T_PREC const * const rdpData   ,    \
        unsigned int   const rnElements,    \
        cudaStream_t         rStream        \
    );
    __INSTANTIATE_TMP(Min,float)
    __INSTANTIATE_TMP(Min,double)
    __INSTANTIATE_TMP(Max,float)
    __INSTANTIATE_TMP(Max,double)
    __INSTANTIATE_TMP(Sum,float)
    __INSTANTIATE_TMP(Sum,double)
    #undef __INSTANTIATE_TMP


    template
    __global__ void cudaKernelCalculateHioError
    <cufftComplex, float>
    (
        cufftComplex const * const rdpgPrime       ,
        float        const * const rdpIsMasked     ,
        unsigned int         const rnData          ,
        bool                 const rInvertMask     ,
        float              * const rdpTotalError   ,
        float              * const rdpnMaskedPixels
    );

    #define __INSTANTIATE_TMP(T_COMPLEX,T_MASK)     \
    template                                        \
    float  cudaCalculateHioError                    \
    <T_COMPLEX, T_MASK>                             \
    (                                               \
        T_COMPLEX const * const rdpData        ,    \
        T_MASK    const * const rdpIsMasked    ,    \
        unsigned int      const rnElements     ,    \
        bool              const rInvertMask    ,    \
        cudaStream_t            rStream        ,    \
        float           * const rpTotalError   ,    \
        float           * const rpnMaskedPixels     \
    );
    __INSTANTIATE_TMP( cufftComplex, float )
    __INSTANTIATE_TMP( cufftComplex, bool )
    __INSTANTIATE_TMP( cufftComplex, unsigned char )


} // namespace cuda
} // namespace algorithms
} // namespace imresh
