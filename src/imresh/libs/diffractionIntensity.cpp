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


#include "diffractionIntensity.hpp"

#include <cmath>     // sqrtf
#ifdef USE_FFTW
#   include <fftw3.h>  // we only need fftw_complex from this and don't want to confuse the compiler if cufftw is being used, so include it here instead of in the header
#else
#   include <cuda_runtime_api.h>
#   include <cufft.h>
#   include "libs/cudacommon.hpp"
#   include "libs/checkCufftError.hpp"
#   include "algorithms/cuda/cudaVectorElementwise.hpp"
#endif
#include "libs/vectorIndex.hpp"


namespace imresh
{
namespace libs
{


#   ifdef USE_FFTW

        void diffractionIntensity
        (
            float * const rIoData,
            unsigned int const rImageWidth,
            unsigned int const rImageHeight
        )
        {
            unsigned int nElements = rImageWidth * rImageHeight;
            /* @ see http://www.fftw.org/doc/Precision.html */
            auto tmp = new fftwf_complex[nElements];

            for ( unsigned i = 0; i < nElements; ++i )
            {
                tmp[i][0] = rIoData[i];
                tmp[i][1] = 0;
            }
            fftwf_plan ft = fftwf_plan_dft_2d(
                rImageHeight /* nRows */, rImageWidth /* nColumns */,
                tmp, tmp, FFTW_FORWARD, FFTW_ESTIMATE );
            fftwf_execute( ft );
            fftwf_destroy_plan( ft );

            for ( unsigned i = 0; i < nElements; ++i )
            {
                const float & re = tmp[i][0]; /* Re */
                const float & im = tmp[i][1]; /* Im */
                const float norm = sqrtf( re*re + im*im );
                rIoData[ i /*fftShiftIndex(i,rSize)*/ ] = norm;
            }

            delete[] tmp;
        }

#   else

        void diffractionIntensity
        (
            float * const rIoData,
            unsigned int const rImageWidth,
            unsigned int const rImageHeight,
            cudaStream_t const rStream,
            bool rAsync
        )
        {
            const unsigned int nElements = rImageWidth * rImageHeight;

            cufftComplex * dpTmp;
            const unsigned int tmpSize = nElements * sizeof( dpTmp[0] );
            CUDA_ERROR( cudaMalloc( (void**)&dpTmp, tmpSize ) );
            /* what we want to do is copy the float input array to the real
             * part of the cufftComplex array and set the imaginary part to 0 */
            CUDA_ERROR( cudaMemsetAsync( dpTmp, 0, tmpSize, rStream ) );
            CUDA_ERROR( cudaMemcpy2DAsync( dpTmp  , sizeof( dpTmp  [0] ),
                                           rIoData, sizeof( rIoData[0] ),
                                           sizeof( rIoData[0] ), nElements,
                                           cudaMemcpyHostToDevice, rStream ) );

            cufftHandle ftPlan;
            CUFFT_ERROR( cufftPlan2d( &ftPlan, rImageHeight /* nRows */, rImageWidth /* nColumns */, CUFFT_C2C ) );
            CUFFT_ERROR( cufftSetStream( ftPlan, rStream ) );
            CUFFT_ERROR( cufftExecC2C( ftPlan, dpTmp, dpTmp, CUFFT_FORWARD ) );
            CUFFT_ERROR( cufftDestroy( ftPlan ) );

            imresh::algorithms::cuda::cudaComplexNormElementwise( dpTmp, dpTmp, nElements, rStream, true );

            CUDA_ERROR( cudaMemcpy2DAsync( rIoData, sizeof( rIoData[0] ),
                                           dpTmp  , sizeof( dpTmp  [0] ),
                                           sizeof( rIoData[0] ), nElements,
                                           cudaMemcpyDeviceToHost, rStream ) );
            if ( not rAsync )
                CUDA_ERROR( cudaStreamSynchronize( rStream ) );
        }

#   endif


} // diffractionIntensity
} // namespace imresh
