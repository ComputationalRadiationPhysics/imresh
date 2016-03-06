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

#include <string>               // std::string
#include <utility>              // std::pair


namespace imresh
{
namespace io
{
namespace writeOutFuncs
{


    /**
     * Just free the pointer and delete the memory.
     *
     * This function only exists for benchmarking purposes, as it's not
     * dependant on the filesystem.
     *
     * @param[in]  _mem pointer to free
     * @param[out] _mem pointer will be set to NULL after freed
     * @param _size ignored
     * @param _filename ignored
     */
    void justFree(
        float * _mem,
        std::pair<unsigned int,unsigned int> const _size,
        std::string const _filname
    );

#   ifdef USE_PNG
        /**
         * Writes the reconstructed image to a PNG file.
         *
         * @param[in] _mem image data in row-major order
         * @param[in] _size image dimensions, order: (width, height), i.e.
         *            (number of columns, number of rows) when thinking in
         *            in matrices
         * @param[in] _filename file name including .png extension and or path.
         *            Note that paths will only work, if all folders do exist.
         */
        void writeOutPNG(
            float * _mem,
            std::pair<unsigned int,unsigned int> const _size,
            std::string const _filename
        );

        /**
         * Writes the reconstructed image to a PNG file and frees data.
         * @see writeOutPNG @see justFree
         */
        void writeOutAndFreePNG(
            float * _mem,
            std::pair<unsigned int,unsigned int> const _size,
            std::string const _filename
        );
#   endif

#   ifdef USE_SPLASH
        /**
         * Write out data using HDF5.
         *
         * This is done using libSplash.
         */
        void writeOutHDF5(
            float * _mem,
            std::pair<unsigned int,unsigned int> const _size,
            std::string const _filename
        );

        /**
         * Write out data using HDF5 and free data.
         *
         * This is done using libSplash.
         */
        void writeOutAndFreeHDF5(
            float * _mem,
            std::pair<unsigned int,unsigned int> const _size,
            std::string const _filename
        );
#   endif


} // namespace writeOutFuncs
} // namespace io
} // namespace imresh
