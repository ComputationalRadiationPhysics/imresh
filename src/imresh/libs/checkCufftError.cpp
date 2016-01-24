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


#include "checkCufftError.hpp"

#include <cstdio>
#include <cassert>
#include <cstdlib>    // EXIT_FAILURE, exit


namespace imresh
{
namespace libs
{


    static const char *
    cufftGetErrorString( cufftResult error )
    {
        switch ( error )
        {
            case CUFFT_SUCCESS:
                return "CUFFT_SUCCESS";
            case CUFFT_INVALID_PLAN:
                return "CUFFT_INVALID_PLAN";
            case CUFFT_ALLOC_FAILED:
                return "CUFFT_ALLOC_FAILED";
            case CUFFT_INVALID_TYPE:
                return "CUFFT_INVALID_TYPE";
            case CUFFT_INVALID_VALUE:
                return "CUFFT_INVALID_VALUE";
            case CUFFT_INTERNAL_ERROR:
                return "CUFFT_INTERNAL_ERROR";
            case CUFFT_EXEC_FAILED:
                return "CUFFT_EXEC_FAILED";
            case CUFFT_SETUP_FAILED:
                return "CUFFT_SETUP_FAILED";
            case CUFFT_INVALID_SIZE:
                return "CUFFT_INVALID_SIZE";
            case CUFFT_UNALIGNED_DATA:
                return "CUFFT_UNALIGNED_DATA";
            case CUFFT_INCOMPLETE_PARAMETER_LIST:
                return "CUFFT_INCOMPLETE_PARAMETER_LIST";
            case CUFFT_INVALID_DEVICE:
                return "CUFFT_INVALID_DEVICE";
            case CUFFT_PARSE_ERROR:
                return "CUFFT_PARSE_ERROR";
            case CUFFT_NO_WORKSPACE:
                return "CUFFT_NO_WORKSPACE";
            case CUFFT_NOT_IMPLEMENTED:
                return "CUFFT_NOT_IMPLEMENTED";
            case CUFFT_LICENSE_ERROR:
                return "CUFFT_LICENSE_ERROR";
            default:
                return "Unknown Cufft Error!";
        }
    }


    void checkCufftError
    ( const cufftResult rValue, const char * file, int line )
    {
        if ( rValue != CUFFT_SUCCESS )
        {
            printf( "CUFFT error in %s line:%i : %s\n",
                    file, line, cufftGetErrorString(rValue) );
            exit( EXIT_FAILURE );
        }
    }


} // namespace libs
} // namespace imresh
