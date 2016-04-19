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
#   include <cuda_to_cupla.hpp>
#   include "libs/cufft_to_cupla.hpp"
#   include "libs/cudacommon.hpp"
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
            float * const rpIoData,
            unsigned int const rImageWidth,
            unsigned int const rImageHeight,
            cudaStream_t const rStream,
            bool rAsync
        )
        {
            using namespace imresh::libs; // mallocCudaArray, CudaKernelConfig
            using namespace imresh::algorithms::cuda;  // cudaComplexNormElementwise

            const unsigned int nElements = rImageWidth * rImageHeight;

            cufftComplex * dpTmp;
            const unsigned int tmpSize = nElements * sizeof( dpTmp[0] );
            mallocCudaArray( &dpTmp, nElements );
            /* what we want to do is copy the float input array to the real
             * part of the cufftComplex array and set the imaginary part to 0 */
            CUDA_ERROR( cudaMemset( dpTmp, 0, 1 ) );
            CUDA_ERROR( cudaMemcpy2DAsync(
                dpTmp,                  /* destination address */
                sizeof( dpTmp[0] ),     /* destination pitch */
                rpIoData,               /* source address */
                sizeof( rpIoData[0] ),  /* source pitch */
                sizeof( rpIoData[0] ),  /* width of matrix in bytes */
                nElements,              /* height of matrix (count of coulmns) */
                cudaMemcpyHostToDevice,
                rStream
            ) );
            /* FT[dpTemp] -> dpTmp */
            /* shorthand for HaLT wrapper */
            auto wcdp /* wrapComplexDevicePointer */ =
                [ &rImageHeight,  &rImageWidth ]( cufftComplex * const & rdp )
                {
                    auto arraySize = types::Vec2
                                     {
                                        rImageHeight /* Ny, nRows */,
                                        rImageWidth  /* Nx, nCols */
                                     };
                    return mem::wrapPtr
                           <
                               true /* is complex */,
                               true /* is device pointer */
                           >( (types::Complex<float> *) rdp, arraySize );
                };
            using PlanForward = FFT_Definition<
                FFT_Kind::Complex2Complex,
                2              , /* dims      */
                float          , /* precision */
                std::true_type , /* inverse   */
                true             /* in-place  */
            >;
            auto dpInOut = PlanForward::wrapInput ( wcdp( dpTmp ) );
            auto fftForward = makeFftPlan( dpInOut );
            /* problem: don't know how to get ftPlan from lifft */
            //#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
            //    cufftSetStream( ftPlan, rStream );
            //#endif
            fftForward( dpInOut );

            cudaComplexNormElementwise( CudaKernelConfig( 0,0,0, rStream ), dpTmp, dpTmp, nElements );
            /* move GPU -> CPU */
            CUDA_ERROR( cudaMemcpy2DAsync(
                rpIoData,               /* destination address */
                sizeof( rpIoData[0] ),  /* destination pitch */
                dpTmp,                  /* source address */
                sizeof( dpTmp[0] ),     /* source pitch */
                sizeof( rpIoData[0] ),  /* width of matrix in bytes */
                nElements,              /* height of matrix (count of coulmns) */
                cudaMemcpyDeviceToHost,
                rStream
            ) );
            if ( not rAsync )
                CUDA_ERROR( cudaStreamSynchronize( rStream ) );
        }

#   endif


} // diffractionIntensity
} // namespace imresh
