/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Philipp Trommler, Maximilian Knespel
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

#pragma once

#include <functional>               // std::function
#include <list>                     // std::list
#include <utility>                  // std::pair
#include <thread>


namespace imresh
{
namespace io
{


    /**
     * Calls imresh::io::addTaskAsync() in a thread.
     *
     * @param _h_mem Pointer to the image data.
     * @param _size Size of the memory to be adressed.
     * @param _writeOutFunc A function pointer (std::function) that will be
     * used to handle the processed data.
     * @param _filename The filename to use to save the processed image. Note
     * that some write out functions will take a file extension (as '.png') and
     * some others may not.
     * @param _numberOfCycles Number of iterations to run shrink wrap for.
     * @param _numberOfHIOCycles Number of iterations to run the initial
     * hybrid input output for.
     * @param _targetError The target error to stop the program when reached.
     */
    void addTask(
        float* _h_mem,
        std::pair<unsigned int,unsigned int> _size,
        std::function<void(float*,std::pair<unsigned int,unsigned int>,
            std::string)> _writeOutFunc,
        std::string _filename,
        unsigned int _numberOfCycles = 20,
        unsigned int _numberOfHIOCycles = 20,
        float _targetError = 0.00001f,
        float _HIOBeta = 0.9f,
        float _intensityCutOffAutoCorel = 0.04f,
        float _intensityCutOff = 0.2f,
        float _sigma0 = 3.0f,
        float _sigmaChange = 0.01f
    );

    /**
     * Initializes the library.
     *
     * This is the _first_ call you should make in order to use this library.
     */
    void taskQueueInit( );

    /**
     * Deinitializes the library.
     *
     * This is the last call you should make in order to clear the library's
     * members.
     */
    void taskQueueDeinit( );


} // namespace io
} // namespace imresh
