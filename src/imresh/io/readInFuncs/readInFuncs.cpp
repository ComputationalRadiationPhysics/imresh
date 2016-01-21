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
#ifdef USE_SPLASH
#   include <splash/splash.h>
#endif
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
#           ifdef IMRESH_DEBUG
                std::cout << "imresh::io::readInFuncs::readTxt(): Successfully read file."
                    << std::endl;
#           endif
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

#   ifdef USE_SPLASH
        std::pair<float*,std::pair<unsigned int,unsigned int>> readHDF5(
            std::string _filename
        )
        {
            splash::SerialDataCollector sdc( 0 );
            splash::DataCollector::FileCreationAttr fCAttr;
            splash::DataCollector::initFileCreationAttr( fCAttr );

            fCAttr.fileAccType = splash::DataCollector::FAT_READ;

            sdc.open( _filename.c_str( ), fCAttr );

            int32_t *ids = NULL;
            size_t num_ids = 0;
            sdc.getEntryIDs( NULL, &num_ids );

            if( num_ids == 0 )
            {
                sdc.close( );
#               ifdef IMRESH_DEBUG
                    std::cout << "imresh::io::readInFuncs::readHDF5(): Error opening empty file."
                        << std::endl;
#               endif
                exit( EXIT_FAILURE );
            }
            else
            {
                ids = new int32_t[num_ids];
                sdc.getEntryIDs( ids, &num_ids );
            }

            splash::DataCollector::DCEntry *entries = NULL;
            size_t num_entries = 0;
            sdc.getEntriesForID( ids[0], NULL, &num_entries );

            if( num_entries == 0 )
            {
                delete[] entries;
                delete[] ids;
                sdc.close( );
#               ifdef IMRESH_DEBUG
                    std::cout << "imresh::io::readInFuncs::readHDF5(): Error opening empty file."
                        << std::endl;
#               endif
                exit( EXIT_FAILURE );
            }
            else
            {
                entries = new splash::DataCollector::DCEntry[num_entries];
                sdc.getEntriesForID( ids[0], entries, &num_entries );
            }

            splash::DataCollector::DCEntry first_entry = entries[0];

            splash::Dimensions dim;
            sdc.read( ids[0], first_entry.name.c_str( ), dim, NULL );

            float* mem = NULL;
            if( dim.getScalarSize( ) == 0 )
            {
                delete[] entries;
                delete[] ids;
                delete[] mem;
                sdc.close( );
#               ifdef IMRESH_DEBUG
                    std::cout << "imresh::io::readInFuncs::readHDF5(): Error opening empty file."
                        << std::endl;
#               endif
                exit( EXIT_FAILURE );
            }
            else
            {
                sdc.read( ids[0], first_entry.name.c_str( ), dim, mem );
            }

            delete[] entries;
            delete[] ids;

#           ifdef IMRESH_DEBUG
                std::cout << "imresh::io::readInFuncs::readHDF5(): Successfully read file."
                    << std::endl;
#           endif
            return { mem, { dim[1], dim[2] } };
        }
#   endif
} // namespace readInFuncs
} // namespace io
} // namespace imresh
