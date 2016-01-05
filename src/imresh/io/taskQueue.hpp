#include <list>
#include <mutex>
#include <thread>

namespace imresh
{
namespace io
{
    std::mutex mtx;
    std::list<cudaEvent_t> eventList;
    std::list<imresh::hal::stream> streamList;

    /*
     * Inserts a new task into the task queue.
     *
     * The task will be added to the next CUDA stream available. This is done
     * asynchronously.
     *
     * @param _h_mem Pointer to the host memory. This has to be pinned
     * memory allocated with cudaMallocHost.
     * @param _size Size of the host data.
     */
    void addTask(
        int* _h_mem,
        int _size
    );

    static void addTaskAsync(
        int* _h_mem,
        int _size
    );

    static void listenForEvents( );
    };
} // namespace io
} // namespace imresh
