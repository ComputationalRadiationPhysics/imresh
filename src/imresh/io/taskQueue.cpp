#include <cuda_runtime.h>           // cudaEvent_t
#include <list>                     // std::list
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

#include <mutex>                    // std::mutex
#include <thread>                   // std::thread

#include "algorithms/cuda/cudaShrinkWrap.h"
#include "io/taskQueue.hpp"
#include "libs/cudacommon.h"

namespace imresh
{
namespace io
{
    void addTask(
        int* _h_mem,
        int _size
    )
    {
        std::thread( imresh::io::addTaskAsync, _h_mem, _size).detach( );
    }


    void addTaskAsync(
        int* _h_mem,
        int _size
    )
    {
        // Lock the mutex so no other thread intermediatly changes the device
        // selection
        imresh::io::mtx.lock( );
        // Device memory pointer
        int* d_mem;
        // "Finished" event
        cudaEvent_t event;
        imresh::io::eventList.push_end( event );
        // Get the next device and stream to use
        auto strm = imresh::hal::streamList.front( );
        imresh::hal::streamList.pop_front( );
        imresh::hal::streamList.push_end( strm );
        auto device = strm.device;
        auto str = strm.str;

        // Select device and copy memory
        CUDA_ERROR( cudaSetDevice( device ) );
        CUDA_ERROR( cudaMalloc( (int**)& d_mem, _size ) );
        CUDA_ERROR( cudaMemcpyAsync( d_mem, _h_mem, _size, cudaMemcpyHostToDevice, str ) );

        // Call shrinkWrap in the selected stream on the selected device.
        imresh::algorithms::cuda::cudaShrinkWrap( /* need new parameters for this */ );

        // Copy memory back
        CUDA_ERROR( cudaMemcpyAsync( _h_mem, d_mem, _size, cudaMemcpyDeviceToHost, str ) );
        CUDA_ERROR( cudaEventCreate( event ) );
        imresh::io::mtx.unlock( );
    }
} // namespace io
} // namespace imresh
