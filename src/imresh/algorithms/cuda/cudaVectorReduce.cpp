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


    extern "C"
    void __implicitelyInstantiateAllImreshCudaVectorReduce( void )
    {
        cudaVectorMin<float>( (float*) NULL, 0, 0 );
        cudaVectorMin<double>( (double*) NULL, 0, 0 );
        cudaVectorMax<float>( (float*) NULL, 0, 0 );
        cudaVectorMax<double>( (double*) NULL, 0, 0 );
        cudaVectorSum<float>( (float*) NULL, 0, 0 );
        cudaVectorSum<double>( (double*) NULL, 0, 0 );

        /* cudaKernelCalculateHioError instantiated by below functions */
        cudaCalculateHioError< cufftComplex, float >(
            (cufftComplex*) NULL, (float*) NULL, 0, false, 0,
            (float*) NULL, (float*) NULL );
        cudaCalculateHioError< cufftComplex, bool >(
            (cufftComplex*) NULL, (bool*) NULL, 0, false, 0,
            (float*) NULL, (float*) NULL );
        cudaCalculateHioError< cufftComplex, unsigned char >(
            (cufftComplex*) NULL, (unsigned char*) NULL, 0, false, 0,
            (float*) NULL, (float*) NULL );
    }


} // namespace cuda
} // namespace algorithms
} // namespace imresh
