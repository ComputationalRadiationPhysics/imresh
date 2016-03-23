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


#include "cudaVectorElementwise.hpp"
#include "cudaVectorElementwise.tpp"


namespace imresh
{
namespace algorithms
{
namespace cuda
{


    #define __INSTANTIATE_TMP( T_COMPLEX, T_PREC )                \
    template                                                      \
    void cudaApplyHioDomainConstraints< T_COMPLEX, T_PREC >       \
    (                                                             \
        libs::CudaKernelConfig  const              rKernelConfig, \
        T_COMPLEX             * const __restrict__ rdpgPrevious , \
        T_COMPLEX       const * const __restrict__ rdpgPrime    , \
        T_PREC          const * const __restrict__ rdpIsMasked  , \
        unsigned int            const              rnElements   , \
        T_PREC                  const              rHioBeta       \
    );
    //__INSTANTIATE_TMP( cufftComplex, float )
    #undef __INSTANTIATE_TMP

    #define __INSTANTIATE_TMP( T_PREC )             \
    float compareCpuWithGpuArray< T_PREC >          \
    (                                               \
        T_PREC const * const __restrict__ rpData,   \
        T_PREC const * const __restrict__ rdpData,  \
        unsigned int const rnElements               \
    );
    //__INSTANTIATE_TMP( float )
    #undef __INSTANTIATE_TMP


} // namespace cuda
} // namespace algorithms
} // namespace imresh
