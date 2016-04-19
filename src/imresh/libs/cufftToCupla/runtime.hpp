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


#include <haLT/FFT.hpp>


namespace cupla
{
namespace cufftToCupla
{


    // CUFFT API function return values
    typedef enum cufftResult_t
    {
        CUFFT_SUCCESS                   = 0x00,
        CUFFT_INVALID_PLAN              = 0x01,
        CUFFT_ALLOC_FAILED              = 0x02,
        CUFFT_INVALID_TYPE              = 0x03,
        CUFFT_INVALID_VALUE             = 0x04,
        CUFFT_INTERNAL_ERROR            = 0x05,
        CUFFT_EXEC_FAILED               = 0x06,
        CUFFT_SETUP_FAILED              = 0x07,
        CUFFT_INVALID_SIZE              = 0x08,
        CUFFT_UNALIGNED_DATA            = 0x09,
        CUFFT_INCOMPLETE_PARAMETER_LIST = 0x0A,
        CUFFT_INVALID_DEVICE            = 0x0B,
        CUFFT_PARSE_ERROR               = 0x0C,
        CUFFT_NO_WORKSPACE              = 0x0D,
        CUFFT_NOT_IMPLEMENTED           = 0x0E,
        CUFFT_LICENSE_ERROR             = 0x0F
    } cufftResult;

    #define CUFFT_FORWARD -1
    #define CUFFT_INVERSE  1

    typedef float cufftReal;
    typedef double cufftDoubleReal;

    typedef cuComplex cufftComplex;
    typedef cuDoubleComplex cufftDoubleComplex;

    // CUFFT supports the following transform types
    typedef enum cufftType_t
    {
        CUFFT_R2C = 0x2a,     // Real to Complex (interleaved)
        CUFFT_C2R = 0x2c,     // Complex (interleaved) to Real
        CUFFT_C2C = 0x29,     // Complex to Complex, interleaved
        CUFFT_D2Z = 0x6a,     // Double to Double-Complex
        CUFFT_Z2D = 0x6c,     // Double-Complex to Double
        CUFFT_Z2Z = 0x69      // Double-Complex to Double-Complex
    } cufftType;


} // namespace cufftToCupla
} // namespace cupla
