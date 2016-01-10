/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Philipp Trommler
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <functional>               // std::function
#ifdef IMRESH_DEBUG
#   include <iostream>              // std::cout, std::endl
#endif
#include <list>                     // std::list
#include <mutex>                    // std::mutex

#include "libs/cudacommon.h"        // CUDA_ERROR

namespace imresh
{
namespace io
{
    struct stream
    {
        int device;
        cudaStream_t str;
    };

    std::list<stream> streamList;
    std::mutex mtx;

    extern "C" void addTaskAsync(
        int* _h_mem,
        int _size,
        std::function<void(int*,int)> _writeOutFunc
    )
    {
        // Lock the mutex so no other thread intermediatly changes the device
        // selection
        mtx.lock( );
        // Device memory pointer
        int* d_mem;
        // Get the next device and stream to use
        auto strm = streamList.front( );
        streamList.pop_front( );
        streamList.push_back( strm );
        auto device = strm.device;
        auto str = strm.str;

        // Select device and copy memory
        CUDA_ERROR( cudaSetDevice( device ) );
        CUDA_ERROR( cudaMalloc( (int**)& d_mem, _size ) );
        CUDA_ERROR( cudaMemcpyAsync( d_mem, _h_mem, _size, cudaMemcpyHostToDevice, str ) );

        // Call shrinkWrap in the selected stream on the selected device.
        //imresh::algorithms::cuda::cudaShrinkWrap( /* need new parameters for this */ );

        // Copy memory back
        CUDA_ERROR( cudaMemcpyAsync( _h_mem, d_mem, _size, cudaMemcpyDeviceToHost, str ) );
        mtx.unlock( );

        _writeOutFunc(_h_mem, _size);
    }

    extern "C" void fillStreamList( )
    {
#       ifdef IMRESH_DEBUG
            std::cout << "imresh::io::fillStreamList(): Starting stream creation."
                << std::endl;
#       endif
        int deviceCount = 0;
        CUDA_ERROR( cudaGetDeviceCount( &deviceCount ) );

        if( deviceCount <= 0 )
        {
#           ifdef IMRESH_DEBUG
                std::cout << "imresh::io::fillStreamList(): No devices found."
                    << std::endl;
#           endif
            return;
        }

        for( int i = 0; i < deviceCount; i++ )
        {
            cudaDeviceProp prop;
            CUDA_ERROR( cudaGetDeviceProperties( &prop, i ) );

            if( prop.multiProcessorCount <= 0 )
            {
#               ifdef IMRESH_DEBUG
                    std::cout << "imresh::io::fillStreamList(): Devices has no multiprocessors."
                        << std::endl;
#               endif
                return;
            }

            for( int j = 0; j < prop.multiProcessorCount; j++ )
            {
                stream str;
                str.device = i;
                CUDA_ERROR( cudaStreamCreate( &str.str ) );
                streamList.push_back( str );
#               ifdef IMRESH_DEBUG
                    std::cout << "imresh::io::fillStreamList(): Created stream "
                        << j << " on device " << i << std::endl;
#               endif
            }
        }
#       ifdef IMRESH_DEBUG
            std::cout << "imresh::io::fillStreamList(): Finished stream creation."
                << std::endl;
#       endif
    }
} // namespace io
} // namespace imresh
