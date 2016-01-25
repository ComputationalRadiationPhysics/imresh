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

#include <cmath>
#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif
#include <cassert>
#include <cstddef>  // NULL


namespace imresh
{
namespace libs
{


    template<class T_PREC>
    int calcGaussianKernel
    (
        const double & rSigma,
        T_PREC * const & rWeights,
        const unsigned & rnWeights,
        const double & rMinAbsoluteError
    )
    {
        /**
         * @todo inverfc, e.g. with minimax (port python version to C/C++)
         * the inverse erfc diverges at 0, this makes it hard to find a
         * a polynomial approximation there, but maybe I could rewrite
         * minimax algorithm to work with \sum a_n/x**n
         * Anyway, the divergence is also bad for the kernel Size. In order
         * to reach floating point single precision of 1e-7 absolute error
         * the kernel size would be: 3.854659 ok, it diverges much slower
         * than I thought
         **/
        //const int nNeighbors = ceil( erfcinv( 2.0*rMinAbsoluteError ) - 0.5 );
        assert( rSigma >= 0 );
        const int nNeighbors = ceil( 2.884402748387961466 * rSigma - 0.5 );
        const int nWeights   = 2*nNeighbors + 1;
        assert( nWeights > 0 );
        if ( rWeights == NULL or (unsigned) nWeights > rnWeights )
            return nWeights;

        double sumWeightings = 0;
        /* Calculate the weightings. I'm not sure, if this is correct.
         * I mean it could be, that the weights are the integrated gaussian
         * values over the pixel interval, but I guess that would force
         * no interpolation. Depending on the interpolation it wouldn't even
         * be pixel value independent anymore, making this useless, so I guess
         * the normal distribution evaluated at -1,0,1 for a kernel size of 3
         * should be correct ??? */
        const double a =  1.0/( sqrt(2.0*M_PI)*rSigma );
        const double b = -1.0/( 2.0*rSigma*rSigma );
        for ( int i = -nNeighbors; i <= nNeighbors; ++i )
        {
            const T_PREC weight = T_PREC( a*exp( i*i*b ) );
            rWeights[nNeighbors+i] = weight;
            sumWeightings += weight;
        }

        /* scale up or down the kernel, so that the sum of the weights will be 1 */
        for ( int i = -nNeighbors; i <= nNeighbors; ++i )
            rWeights[nNeighbors+i] /= sumWeightings;

        return nWeights;
    }


    /* Explicitely instantiate certain template arguments to make an object
     * file. Furthermore this saves space, as we don't need to write out the
     * data types of all functions to instantiate */

    template int calcGaussianKernel<float>
    (
        const double & rSigma,
        float * const & rWeights,
        const unsigned & rnWeights,
        const double & rMinAbsoluteError
    );
    template int calcGaussianKernel<double>
    (
        const double & rSigma,
        double * const & rWeights,
        const unsigned & rnWeights,
        const double & rMinAbsoluteError
    );


} // namespace libs
} // namespace imresh
