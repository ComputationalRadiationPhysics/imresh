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


#include "vectorReduce.hpp"
#include "vectorReduce.tpp"


namespace benchmark
{
namespace imresh
{
namespace algorithms
{


    SumFunctor<float > sumFunctorf;
    MinFunctor<float > minFunctorf;
    MaxFunctor<float > maxFunctorf;
    SumFunctor<double> sumFunctord;
    MinFunctor<double> minFunctord;
    MaxFunctor<double> maxFunctord;


    /* explicit template instantiations */


    #define INSTANTIATE_TMP( NAME )     \
    template                            \
    float cudaVectorMax##NAME<float>    \
    (                                   \
        float const * const rdpData,    \
        unsigned int const rnElements,  \
        cudaStream_t rStream            \
    );
    INSTANTIATE_TMP( GlobalAtomic2 )
    INSTANTIATE_TMP( GlobalAtomic )
    INSTANTIATE_TMP( SharedMemory )
    INSTANTIATE_TMP( SharedMemoryWarps )
    #undef INSTANTIATE_TMP

    template
    __global__ void cudaKernelCalculateHioErrorBitPacked<cufftComplex>
    (
        cufftComplex const * const __restrict__ rdpgPrime,
        uint32_t     const * const __restrict__ rdpIsMasked,
        unsigned int const rnData,
        float * const __restrict__ rdpTotalError,
        float * const __restrict__ rdpnMaskedPixels
    );

    template
    float cudaCalculateHioErrorBitPacked<cufftComplex>
    (
        cufftComplex const * const rdpData,
        uint32_t  const * const rdpIsMasked,
        unsigned int const rnElements,
        bool const rInvertMask,
        cudaStream_t rStream,
        float * const rpTotalError,
        float * const rpnMaskedPixels
    );


} // namespace algorithms
} // namespace imresh
} // namespace benchmark
