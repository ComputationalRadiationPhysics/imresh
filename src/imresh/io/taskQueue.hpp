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
#include <list>                     // std::list
#include <string>                   // std::string
#include <utility>                  // std::pair

namespace imresh
{
namespace io
{
    extern "C"
    {
        void addTaskAsync(
            float* _h_mem,
            std::pair<unsigned int,unsigned int> _size,
            std::function<void(float*,std::pair<unsigned int,unsigned int>,
                std::string)> _writeOutFunc,
            std::string _filename
        );

        void fillStreamList( );
    }


    class taskQueue
    {
    public:
        /**
         * Standard constructor.
         */
        taskQueue( );

        /**
         * Inserts a new task into the task queue.
         *
         * The task will be added to the next CUDA stream available. This is done
         * asynchronously.
         *
         * @param _h_mem Pointer to the host memory. This has to be pinned
         * memory allocated with cudaMallocHost.
         * @param _size Size of the host data.
         * @param _writeOutFunc The function to use to write the processed data.
         */
        void addTask(
            float* _h_mem,
            std::pair<unsigned int,unsigned int> _size,
            std::function<void(float*,std::pair<unsigned int,unsigned int>,
                std::string)> _writeOutFunc,
            std::string _filename = "imresh"
        );
    };
} // namespace io
} // namespace imresh
