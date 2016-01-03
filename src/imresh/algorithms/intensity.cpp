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


#include <vector>


namespace imresh
{
namespace algorithms
{

    /**
     * shifts an n-dimensional index by half the size
     *
     * The use case is the FFT where the 0 frequency is at the 0th array entry
     * but we want it in the center, in all dimensions.
     *
     * @param[in] rLinearIndex simple linear index which should be in
     *            [0,product(rDim))
     * @param[in] rDim the size of each dimension
     **/
    unsigned fftShiftIndex
    (
        const unsigned & rLinearIndex,
        const std::vector<unsigned> & rDim
    )
    {
        using namespace imresh::algorithms;

        std::vector<unsigned> vectorIndex =
            convertLinearToVectorIndex( rLinearIndex, rDim );
        for ( unsigned i = 0; i < rDim.size(); ++i )
        {
            vectorIndex[i] += rDim.size() / 2;
            vectorIndex[i] %= rDim.size();
        }
        return convertVectorToLinearIndex( vectorIndex, rDim );
    }

    /**
     * Calculates the diffraction intensity of an object function
     *
     * @see https://en.wikipedia.org/wiki/Diffraction#General_aperture
     * Because there are different similar applications no constant physical
     * factors will be applied here, instead a simple fourier transform
     * followed by a squared norm will be used.
     *
     * This function also shifts the frequency, so that frequency 0 is in the
     * middle like it would be for a normal measurement.
     *
     * E.g. a rectangular box becomes a kind of checkerboard pattern with
     * decreasing maxima intensity:
     *
     * @verbatim
     *        +------------+      +------------+          +------------+
     *        |            |      |78 ++  ++ 56|          |     --     |
     *        |            |      |o> ''  '' <o|          | .. <oo> .. |
     *        |     #      |  FT  |-          -| fftshift | ++ 1234 ++ |
     *        |     #      |  ->  |-          -|  ----->  | ++ 5678 ++ |
     *        |            |      |o> ..  .. <o|          | '' <oo> '' |
     *        |            |      |34 ++  ++ 12|          |     --     |
     *        +------------+      +------------+          +------------+
     *                           k=0         k=N-1              k=0
     * @endverbatim
     *
     * @param[in]  rIoData real valued object which is to be transformed
     * @param[out] rIoData real valued diffraction intensity
     * @param[in]  rDimensions vector containing the dimensions of rIoData.
     *             the number of dimensions i.e. the size of the vector can be
     *             anything > 0
     **/
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
        fftwf_complex * tmp = fftwf_alloc_complex( nElementsReduced );

        /* forward fourier transform the original image, i.e. negative
         * coefficients @see http://www.fftw.org/fftw3_doc/The-1d-Discrete-Fourier-Transform-_0028DFT_0029.html#The-1d-Discrete-Fourier-Transform-_0028DFT_0029 */
        fftwf_plan planForward = fftwf_plan_dft_r2c( rDim.size(), &rDim[0],
            rIoData, tmp, FFTW_ESTIMATE );
        /* FFTW_FORWARD not needed, automatically assumed for r2c */
        fftwf_execute( planForward );

        /* strip fourier transformed real image of it's phase (measurement) */
        for ( unsigned iRow = 0; iRow < nElements / lastDim; ++iRow )
        for ( unsigned iCol = 0; iCol < reducedLastDim; ++iCol )
        {
            const float & re = F[i][0]; /* Re */
            const float & im = F[i][1]; /* Im */
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
            const unsigned i0 = fftShiftIndex( iRow*lastDim + iCol );
            const unsigned i1 = fftShiftIndex( iRow*lastDim + lastDim-iCol );
            rIoData[i0]] = norm;
            rIoData[i1]] = norm;
        }

        fftwf_destroy_plan( planForward );
        fftwf_free( tmp );
    }


} // namespace algorithms
} // namespace imresh
