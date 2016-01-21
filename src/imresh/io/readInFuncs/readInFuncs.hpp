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

#include <string>               // std::string
#include <utility>              // std::pair

namespace imresh
{
namespace io
{
namespace readInFuncs
{
    /**
     * Simple function for reading txt files.
     *
     * They need to store their values as a 2D matrix with spaces as delimiters.
     */
    std::pair<float*,std::pair<unsigned int,unsigned int>> readTxt(
        std::string _filename
    );

#   ifdef USE_SPLASH
        std::pair<float*,std::pair<unsigned int,unsigned int>> readHDF5(
            std::string _filename
        );
#   endif
} // namespace readInFuncs
} // namespace io
} // namespace imresh
