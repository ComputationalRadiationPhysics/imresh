/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Maximilian Knespel
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


#pragma once


/* only load redefinitions for cuComplex and FFT_C2C,... if using HaLT
 * without NVCC */
#if ! defined( _CUFFT_H_ ) && ! defined( __CUDACC__ )
#   include "cufftToCupla/cuComplex.hpp"
#   include "cufftToCupla/runtime.hpp"
#endif

#include <haLT/FFT.hpp>
using namespace haLT;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#   include "haLT/libraries/cuFFT/cuFFT.hpp"
    using FFT_Backend = libraries::cuFFT::CuFFT<>;
    namespace cupla {
    namespace cufftToCupla {
        constexpr bool devicePointerExists = true;
    } }
#else
#   include "haLT/libraries/fftw/FFTW.hpp"
    using FFT_Backend = libraries::fftw::FFTW<>;
    namespace cupla {
    namespace cufftToCupla {
        constexpr bool devicePointerExists = false;
    } }
#endif
#define makeFftPlan(...) makeFFT< FFT_Backend >( __VA_ARGS__ )

namespace cupla {
namespace cufftToCupla {

    /* shorthand for HaLT wrapper */
    inline auto wrapComplexDevicePointer(
        cufftComplex * const & rdp,
        unsigned int const rImageHeight,
        unsigned int const rImageWidth
    ) -> decltype( haLT::mem::wrapPtr<
             true /* is complex */,
             devicePointerExists /* is device pointer */
         >( (haLT::types::Complex<float> *) rdp, haLT::types::Vec2{1,1} ) )
    {
        return haLT::mem::wrapPtr<
                   true /* is complex */,
                   devicePointerExists /* is device pointer */
               >( (haLT::types::Complex<float>*) rdp,
                   haLT::types::Vec2{ rImageHeight /* Ny, nRows */,
                                      rImageWidth  /* Nx, nCols */ } );
    };

    /* shorthand for HaLT wrapper */
    inline auto wrapComplexDevicePointer(
        cufftComplex * const & rdp,
        unsigned int const rnValues
    ) -> decltype( haLT::mem::wrapPtr<
             true /* is complex */,
             devicePointerExists /* is device pointer */
         >( (haLT::types::Complex<float> *) rdp, haLT::types::Vec1{1} ) )
    {
        return haLT::mem::wrapPtr<
                   true /* is complex */,
                   devicePointerExists /* is device pointer */
               >( (haLT::types::Complex<float>*) rdp,
                   haLT::types::Vec1{ rnValues } );
    };

} // cufftToCupla
} // cupla

using namespace cupla::cufftToCupla;
