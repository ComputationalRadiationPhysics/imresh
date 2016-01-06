#ifdef IMRESH_DEBUG
#   include <iostream>
#endif
#include <list>
#include <mutex>

#include "libs/cudacommon.h"

namespace imresh
{
namespace io
{
    struct stream
    {
        int device;
        cudaStream_t str;
    };

    std::list<cudaEvent_t> eventList;
    std::list<stream> streamList;
    std::mutex mtx;

    extern "C" void addTaskAsync(
        int* _h_mem,
        int _size
    )
    {
        // Lock the mutex so no other thread intermediatly changes the device
        // selection
        mtx.lock( );
        // Device memory pointer
        int* d_mem;
        // "Finished" event
        cudaEvent_t event;
        eventList.push_back( event );
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
        CUDA_ERROR( cudaEventCreate( &event ) );
        mtx.unlock( );
    }

    extern "C" void fillStreamList( )
    {
#       ifdef IMRESH_DEBUG
            std::cout << "imresh::hal::fillStreamList(): Starting stream \
                creation." << std::endl;
#       endif
        int deviceCount;
        CUDA_ERROR( cudaGetDeviceCount( &deviceCount ) );

        for( int i = 0; i < deviceCount; i++ )
        {
            cudaDeviceProp prop;
            CUDA_ERROR( cudaGetDeviceProperties( &prop, i ) );

            for( int j = 0; j < prop.multiProcessorCount; j++ )
            {
                stream str;
                str.device = i;
                CUDA_ERROR( cudaStreamCreate( &str.str ) );
                streamList.push_back( str );
#               ifdef IMRESH_DEBUG
                    std::cout << "imresh::hal::fillStreamList(): Created stream "
                        << j << " on device " << i << std::endl;
#               endif
            }
        }
#       ifdef IMRESH_DEBUG
            std::cout << "imresh::hal::fillStreamList(): Finished stream \
                creation." << std::endl;
#       endif
    }
} // namespace io
} // namespace imresh
