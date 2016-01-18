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

#include <algorithm>            // std::count_if
#include <fstream>              // open
#include <iostream>             // std::cout
#include <list>                 // std::list
#include <string>               // std::string
#include <utility>              // std::pair

#include "io/readInFuncs/readInFuncs.hpp"

namespace imresh
{
namespace io
{
namespace readInFuncs
{
    std::pair<float*,std::pair<unsigned int,unsigned int>> readTxt(
        std::string _filename
    )
    {
        std::ifstream file;
        file.open( _filename, std::ios::in );

        if( file.is_open( ) )
        {
            std::list<std::string> strList;
            std::string str;
            std::getline( file, str );

            // First, determine the length of a line, e.g. the x dimension
            unsigned int xDim = std::count_if( str.begin( ), str.end( ),
                [ ]( unsigned char c ){ return std::isspace( c ); } ) + 1;

            // Then go back to the beginning of the file
            file.seekg( 0, std::ios::beg );

            // Now iterate through the whole file
            while( getline( file, str, ' ' ) )
            {
                strList.push_back( str );
            }

            // Now determine the y dimension
            unsigned int yDim = strList.size( ) / xDim;

            // And last, convert and store the list into a array
            float retArray[strList.size( )];
            for( auto i = 0; i < strList.size( ); i++ )
            {
                retArray[i] = std::stof( strList.front( ) );
                strList.pop_front( );
            }

            auto ret = std::make_pair( (float*) retArray, std::make_pair( xDim, yDim ) );
            return ret;
        }
        else
        {
#           ifdef IMRESH_DEBUG
                std::cout << "imresh::io::readInFuncs::readTxt(): Error opening file."
                    << std::endl;
#           endif
            exit( EXIT_FAILURE );
        }
    }
} // namespace readInFuncs
} // namespace io
} // namespace imresh
