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


    extern "C"
    void __implicitelyInstantiateAllImreshCudaVectorElementwise( void )
    {
        CUPLA_KERNEL
            ( cudaKernelApplyHioDomainConstraints<cufftComplex, float> )
            (1,1)
            ( (cufftComplex*) NULL, (cufftComplex*) NULL, (float*) NULL, 0, 0);

        CUPLA_KERNEL
            ( cudaKernelCopyToRealPart<cufftComplex, float> )
            (1,1)
            ( (cufftComplex*) NULL, (float*) NULL, 0);

        CUPLA_KERNEL
            ( cudaKernelCopyFromRealPart<float, cufftComplex> )
            (1,1)
            ( (float*) NULL, (cufftComplex*) NULL, 0);

        /* implicitely instantiated by cudaComplexNormElementwise */
        /*
        CUPLA_KERNEL
            ( cudaKernelComplexNormElementwise<float, cufftComplex> )
            (1,1)
            ( (float*) NULL, (cufftComplex*) NULL, 0 );
        CUPLA_KERNEL
            ( cudaKernelComplexNormElementwise<cufftComplex, cufftComplex> )
            (1,1)
            ( (cufftComplex*) NULL, (cufftComplex*) NULL, 0 );
        */

        cudaComplexNormElementwise< float, cufftComplex >(
            (float*) NULL, (cufftComplex*) NULL, 0, 0, false );
        cudaComplexNormElementwise< cufftComplex, cufftComplex >(
            (cufftComplex*) NULL, (cufftComplex*) NULL, 0, 0, false );

        CUPLA_KERNEL
            ( cudaKernelApplyComplexModulus< cufftComplex, float > )
            (1,1)
            ( (cufftComplex*) NULL, (cufftComplex*) NULL, (float*) NULL, 0 );

        CUPLA_KERNEL
            ( cudaKernelCutOff<float> )
            (1,1)
            ( (float*) NULL, 0, 0.0f, 0.0f, 0.0f );
        CUPLA_KERNEL
            ( cudaKernelCutOff<double> )
            (1,1)
            ( (double*) NULL, 0, 0.0, 0.0, 0.0 );
    }


} // namespace cuda
} // namespace algorithms
} // namespace imresh
