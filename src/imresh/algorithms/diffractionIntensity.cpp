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


#include "diffractionIntensity.h"


namespace imresh
{
namespace algorithms
{


    template<class T>
    std::ostream & operator<<
    (
        std::ostream & rOut,
        const std::vector<T> & rVectorToPrint
    )
    {
        rOut << "{ ";
        for ( const auto & elem : rVectorToPrint )
            rOut << elem << " ";
        rOut << "}";
        return rOut;
    }

    unsigned fftShiftIndex
    (
        const unsigned & rLinearIndex,
        const std::vector<unsigned> & rDim
    )
    {
        using namespace imresh::algorithms;

        std::vector<unsigned> vectorIndex =
            convertLinearToVectorIndex( rLinearIndex, rDim );
        //std::cout << "   lini=" << rLinearIndex << " -> " << vectorIndex << "\n";
        for ( unsigned i = 0; i < rDim.size(); ++i )
        {
            vectorIndex[i] += rDim[i] / 2;
            vectorIndex[i] %= rDim[i];
        }
        return convertVectorToLinearIndex( vectorIndex, rDim );
    }


    void diffractionIntensity
    (
        float * const & rIoData,
        const std::vector<unsigned> & rDim
    )
    {
        unsigned nElements = 1;
        for ( const auto & dim : rDim )
            nElements *= dim;
        const unsigned & lastDim = rDim[ rDim.size()-1 ];
        const unsigned reducedLastDim = lastDim/2+1;
        const unsigned nElementsReduced = nElements / lastDim * reducedLastDim;
        /* @ see http://www.fftw.org/doc/Precision.html */
        fftwf_complex * tmp = fftwf_alloc_complex( nElementsReduced );

        /* forward fourier transform the original image, i.e. negative
         * coefficients @see http://www.fftw.org/fftw3_doc/The-1d-Discrete-Fourier-Transform-_0028DFT_0029.html#The-1d-Discrete-Fourier-Transform-_0028DFT_0029 */
        fftwf_plan planForward = fftwf_plan_dft_r2c( rDim.size(), (int*) /*dangerous*/ &rDim[0],
            rIoData, tmp, FFTW_ESTIMATE );
        /* FFTW_FORWARD not needed, automatically assumed for r2c */
        fftwf_execute( planForward );

        /* strip fourier transformed real image of it's phase (measurement) */
        memset( rIoData, 0, nElements*sizeof( rIoData[0] ) );
        for ( unsigned iRow = 0; iRow < nElements / lastDim; ++iRow )
        for ( unsigned iCol = 0; iCol < reducedLastDim; ++iCol )
        {
            const unsigned i = iRow*reducedLastDim + iCol;
            const float & re = tmp[i][0]; /* Re */
            const float & im = tmp[i][1]; /* Im */
            const float norm = sqrtf( re*re + im*im );
            /**
             * calculate 2nd index position because of the symmetry of tmp:
             *   @f[ f(-\vec{x}) = -\overline{f(\vec{x})} @f]
             * for the absolute value squared we need this means:
             *   @f[ |f(-\vec{x})|^2 = |f(\vec{x})|^2 @f]
             * a multi-dimensional transform can be seen as a compositon of
             * one dimension fourier transforms. Only the first transformation
             * i.e. dimension will have the above symmetry. With libfftw3
             * this seems to be lastDim, because that's the dimension which is
             * only half the full size.
             * The result is in row major form, meaning lastDim lies contiguous
             * in memory.
             **/
            //const unsigned i0 = fftShiftIndex( iRow*lastDim + iCol, rDim );
            //const unsigned i1 = fftShiftIndex( iRow*lastDim + lastDim-1-iCol,
            const unsigned i0 = iRow*lastDim + iCol;
            rIoData[i0] = norm;
            if ( iCol != 0 )
            {
                const unsigned i1 = iRow*lastDim + lastDim-iCol;
                rIoData[i1] = norm;
            }

            //std::cout << "iRow=" << iRow << ", iCol=" << iCol
            //          << " => i0=" << i0 << ", i1=" << i1 << "\n";
        }

        fftwf_destroy_plan( planForward );
        fftwf_free( tmp );
    }


} // namespace algorithms
} // namespace imresh
