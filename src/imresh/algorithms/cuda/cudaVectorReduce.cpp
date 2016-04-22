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

    /* explicit instantiations */
    #include "libs/alpaka_T_ACC.hpp"
    #define inline

    #define INSTANTIATE_TMP( cudaReduceFunc, T_PREC )   \
    template                                            \
    T_PREC cudaReduceFunc<T_PREC>                       \
    (                                                   \
        CudaKernelConfig rKernelConfig,                 \
        T_PREC const * rdpData,                         \
        unsigned int rnElements                         \
    );
    INSTANTIATE_TMP( cudaVectorMin, float  );
    INSTANTIATE_TMP( cudaVectorMin, double );
    INSTANTIATE_TMP( cudaVectorMin, int    );
    INSTANTIATE_TMP( cudaVectorMax, float  );
    INSTANTIATE_TMP( cudaVectorMax, double );
    INSTANTIATE_TMP( cudaVectorMax, int    );
    INSTANTIATE_TMP( cudaVectorSum, float  );
    INSTANTIATE_TMP( cudaVectorSum, double );
    #undef INSTANTIATE_TMP


    #undef inline
    #undef T_ACC


} // namespace cuda
} // namespace algorithms
} // namespace imresh
