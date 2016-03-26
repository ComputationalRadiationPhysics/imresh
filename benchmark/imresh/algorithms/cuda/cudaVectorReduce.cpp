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

#include "cudaVectorReduce.hpp"
#include "cudaVectorReduce.tpp"

#include <cassert>
#include <cstdio>
#include <cstdint>    // uint64_t
#include <limits>     // numeric_limits
#include <cuda_to_cupla.hpp>     // atomicCAS, atomicAdd
#include <cufft_to_cupla.hpp>    // cufftComplex, cufftDoubleComplex
#include "libs/cudacommon.hpp"
/**
 * Gives only compile errors, e.g.
 *    ptxas fatal   : Unresolved extern function '_ZN6imresh10algorithms4cuda10SumFunctorIfEclEff'
 * so I justd copy-pasted the functors here ...
 **/
//#include "algorithms/cuda/cudaVectorReduce.hpp" // maxFunctor, atomicFunc


namespace benchmark
{
namespace imresh
{
namespace algorithms
{
namespace cuda
{


    SumFunctor<float > sumFunctorf;
    MinFunctor<float > minFunctorf;
    MaxFunctor<float > maxFunctorf;
    SumFunctor<double> sumFunctord;
    MinFunctor<double> minFunctord;
    MaxFunctor<double> maxFunctord;

    /* explicit template instantiations */

    #define INSTANTIATE_TMP( NAME)          \
    template                                \
    float cudaVectorMax##NAME<float>        \
    (                                       \
        CudaKernelConfig rKernelConfig,     \
        float const * const rdpData,        \
        unsigned int const rnElements       \
    );
    INSTANTIATE_TMP( GlobalAtomic2 )
    INSTANTIATE_TMP( GlobalAtomic )
    INSTANTIATE_TMP( SharedMemory )
    INSTANTIATE_TMP( SharedMemoryWarps )
    #undef INSTANTIATE_TMP


    template
    float cudaCalculateHioErrorBitPacked<cufftComplex>
    (
        CudaKernelConfig           rKernelConfig,
        cufftComplex const * const rdpData      ,
        uint32_t     const * const rdpIsMasked  ,
        unsigned int         const rnElements   ,
        bool                 const rInvertMask  ,
        float              * const rpTotalError ,
        float              * const rpnMaskedPixels
    );


} // namespace cuda
} // namespace algorithms
} // namespace imresh
} // namespace benchmark
