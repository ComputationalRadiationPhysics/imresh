#include <list>
#include <mutex>

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
        auto strm = this->streamList.front( );
        imresh::io::streamList.pop_front( );
        imresh::io::streamList.push_end( strm );
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
