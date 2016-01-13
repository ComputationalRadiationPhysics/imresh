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
#include <string>                   // std::string
#include <thread>                   // std::thread
#include <utility>                  // std::pair

#include "io/taskQueue.hpp"

namespace imresh
{
namespace io
{
    taskQueue::taskQueue( )
    {
        this->m_threadPoolMaxSize = fillStreamList( );
    }

    taskQueue::~taskQueue( )
    {
        while( this->m_threadPool.size( ) > 0 )
        {
            this->m_threadPool.front( ).join( );
            this->m_threadPool.pop_front( );
        }
    }

    void taskQueue::addTask(
        float* _h_mem,
        std::pair<unsigned int,unsigned int> _size,
        std::function<void(float*,std::pair<unsigned int,unsigned int>,
            std::string)> _writeOutFunc,
        std::string _filename
    )
    {
        while( this->m_threadPool.size( ) >= this->m_threadPoolMaxSize )
        {
            this->m_threadPool.front( ).join( );
            this->m_threadPool.pop_front( );
        }

        this->m_threadPool.push_back( std::thread( addTaskAsync, _h_mem, _size,
            _writeOutFunc, _filename ) );
    }
} // namespace io
} // namespace imresh
