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
#include <utility>                  // std::pair

#include "algorithms/cuda/cudaShrinkWrap.h"
#include "libs/cudacommon.h"        // CUDA_ERROR

namespace imresh
{
namespace io
{
    /**
     * Struct containing a CUDA stream with it's associated device.
     */
    struct stream
    {
        int device;
        cudaStream_t str;
    };

    /**
     * List where all streams are stored as imresh::io::stream structs.
     */
    std::list<stream> streamList;
    /**
     * Mutex to coordinate device usage.
     */
    std::unique_lock<std::mutex> mtx;

    /**
     * Function to add a image processing task to the queue.
     *
     * This is called from taskQueue::addTask() as a thread to prevent blocking
     * and to ensure that all streams are filled with work. It selects the least
     * recently used stream from the streamList and fills it with new work (FIFO).
     *
     * @param _h_mem Pointer to the image data.
     * @param _size Size of the memory to be adressed.
     * @param _writeOutFunc A function pointer (std::function) that will be
     * used to handle the processed data.
     */
    extern "C" void addTaskAsync(
        float* _h_mem,
        std::pair<unsigned int,unsigned int> _size,
        std::function<void(float*,std::pair<unsigned int,unsigned int>,
            std::string)> _writeOutFunc,
        std::string _filename
    )
    {
        // Lock the mutex so no other thread intermediatly changes the device
        // selection
        mtx.lock( );
        // Get the next device and stream to use
        auto strm = streamList.front( );
        streamList.pop_front( );
        streamList.push_back( strm );
        auto device = strm.device;
        auto str = strm.str;

        // Select device and copy memory
        CUDA_ERROR( cudaSetDevice( device ) );

        // Call shrinkWrap in the selected stream on the selected device.
        imresh::algorithms::cuda::shrinkWrap( _h_mem, _size, str );

        // Copy memory back
        mtx.unlock( );

        _writeOutFunc( _h_mem, _size, _filename );
    }

    /**
     * This function adds all streams to the stream list.
     *
     * To achieve that it iterates over all available devices and creates one
     * stream for each multiprocessor on each device. All these streams are
     * stored in the streamList as imresh::io::stream objects. If no streams are
     * found, the program aborts.
     */
    extern "C" int fillStreamList( )
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
                std::cout << "imresh::io::fillStreamList(): No devices found. Aborting."
                    << std::endl;
#           endif
            exit( EXIT_FAILURE );
        }

        for( int i = 0; i < deviceCount; i++ )
        {
            cudaDeviceProp prop;
            CUDA_ERROR( cudaGetDeviceProperties( &prop, i ) );

            if( prop.multiProcessorCount <= 0 )
            {
#               ifdef IMRESH_DEBUG
                    std::cout << "imresh::io::fillStreamList(): Devices has no multiprocessors. Aborting."
                        << std::endl;
#               endif
                exit( EXIT_FAILURE );
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

        return streamList.size( );
    }
} // namespace io
} // namespace imresh
