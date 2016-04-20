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


#include "taskQueue.hpp"

#include <functional>               // std::function
#ifdef IMRESH_DEBUG
#   include <iostream>              // std::cout, std::endl
#endif
#include <list>                     // std::list
#include <mutex>                    // std::mutex
#include <thread>                   // std::thread
#include <utility>                  // std::pair, forward
#include <cassert>

#include "algorithms/cuda/cudaShrinkWrap.hpp"
#include "libs/cudacommon.hpp"          // CUDA_ERROR
#include "readInFuncs/readInFuncs.hpp"  // ImageDimensions


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
     * Mutex to coordinate device usage.
     */
    std::mutex mtx;
    /**
     * List where all streams are stored as imresh::io::stream structs.
     */
    std::list<stream> streamList;
    /**
     * List to store all created threads.
     */
    std::list<std::thread> threadPool;
    /**
     * Maximum size of the thread pool.
     *
     * This is determined while imresh::io::fillStreamList() as the number of
     * available streams.
     */
    unsigned int threadPoolMaxSize = 0;

    /**
     * Function to add a image processing task to the queue.
     *
     * This is called from taskQueue::addTask() as a thread to prevent blocking
     * and to ensure that all streams are filled with work. It selects the least
     * recently used stream from the streamList and fills it with new work (FIFO).
     *
     * A mutex ensures the correct work balancing over the CUDA streams.
     * However, this mutex doesn't include the call to the write out function.
     * If you need your write out function to be thread safe, you'll have to
     * use your own lock mechanisms inside of this function.
     *
     * @see addTask
     */
    void addTaskAsync(
        float * _h_mem,
        std::pair<unsigned int,unsigned int> _size,
        std::function<void(float *,std::pair<unsigned int,unsigned int>,
            std::string)> _writeOutFunc ,
        std::string _filename           ,
        unsigned int _numberOfCycles    ,
        unsigned int _numberOfHIOCycles ,
        float _targetError              ,
        float _HIOBeta                  ,
        float _intensityCutOffAutoCorel ,
        float _intensityCutOff          ,
        float _sigma0                   ,
        float _sigmaChange
    )
    {
        // Lock the mutex so no other thread intermediatly changes the device
        // selection
        mtx.lock( );
        // Get the next device and stream to use
        auto strm = streamList.front( );
        streamList.pop_front( );
        streamList.push_back( strm );
        mtx.unlock( );
        auto device = strm.device;
        auto str = strm.str;

        // Select device and copy memory
        CUDA_ERROR( cudaSetDevice( device ) );

        cudaDeviceProp prop;
        CUDA_ERROR( cudaGetDeviceProperties( &prop, device ) );
        unsigned int const nThreadsPerBlock = 256;
        /* oversubscribe the GPU by a factor of 2 to account for cudaMalloc
         * and cudaMemcpy stalls */
        unsigned int const nBlocks = 2*prop.maxThreadsPerMultiProcessor / nThreadsPerBlock;

        // Call shrinkWrap in the selected stream on the selected device.
        imresh::algorithms::cuda::cudaShrinkWrap( _h_mem,
                                              _size.first,
                                              _size.second,
                                              str,
                                              nBlocks,
                                              nThreadsPerBlock,
                                              _numberOfCycles,
                                              _targetError,
                                              _HIOBeta,
                                              _intensityCutOffAutoCorel,
                                              _intensityCutOff,
                                              _sigma0,
                                              _sigmaChange,
                                              _numberOfHIOCycles );

        _writeOutFunc( _h_mem, _size, _filename );
    }

    void addTask(
        float * _h_mem,
        std::pair<unsigned int,unsigned int> _size,
        std::function<void(float *,std::pair<unsigned int,unsigned int>,
            std::string)> _writeOutFunc ,
        std::string _filename           ,
        unsigned int _numberOfCycles    ,
        unsigned int _numberOfHIOCycles ,
        float _targetError              ,
        float _HIOBeta                  ,
        float _intensityCutOffAutoCorel ,
        float _intensityCutOff          ,
        float _sigma0                   ,
        float _sigmaChange
    )
    {
        assert( threadPoolMaxSize > 0 and "Did you make a call to taskQueueInit?" );

        while( threadPool.size( ) >= threadPoolMaxSize )
        {
            if ( threadPool.front( ).joinable( ) )
                threadPool.front( ).join( );
            else
            {
            }
            threadPool.pop_front( );
        }

        /* start new thread and save thread id in threadPool */
        threadPool.push_back( std::thread( addTaskAsync, _h_mem,
                                                         _size,
                                                         _writeOutFunc,
                                                         _filename,
                                                         _numberOfCycles,
                                                         _numberOfHIOCycles,
                                                         _targetError,
                                                         _HIOBeta,
                                                         _intensityCutOffAutoCorel,
                                                         _intensityCutOff,
                                                         _sigma0,
                                                         _sigmaChange ) );

    }

    /**
     * This function adds all streams to the stream list.
     *
     * To achieve that it iterates over all available devices and creates one
     * stream for each multiprocessor on each device. All these streams are
     * stored in the streamList as imresh::io::stream objects. If no streams are
     * found, the program aborts.
     */
    unsigned fillStreamList( )
    {
#       ifdef IMRESH_DEBUG
            std::cout << "imresh::io::fillStreamList(): Starting stream creation."
                << std::endl;
#       endif
        int deviceCount = 0;
        CUDA_ERROR( cudaGetDeviceCount( & deviceCount ) );

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
            CUDA_ERROR( cudaSetDevice( i ) );

            int const multiProcessorCount = 3;

            for( int j = 0; j < multiProcessorCount; j++ )
            {
                stream str;
                str.device = i;
                CUDA_ERROR( cudaStreamCreate( & str.str ) );
                streamList.push_back( str );
#               ifdef IMRESH_DEBUG
                    std::cout << "imresh::io::fillStreamList(): Created stream "
                        << j << " on device " << i << std::endl;
#               endif
            }
        }

        return streamList.size( );
    }

    void taskQueueInit( )
    {
        threadPoolMaxSize = fillStreamList( );
    }

    void taskQueueDeinit( )
    {
        threadPoolMaxSize = 0;

        while( threadPool.size( ) > 0 )
        {
            threadPool.front( ).join( );
            threadPool.pop_front( );
        }

        while( streamList.size( ) > 0 )
        {
            CUDA_ERROR( cudaStreamDestroy( streamList.front( ).str ) );
            streamList.pop_front( );
        }
    }

} // namespace io
} // namespace imresh
