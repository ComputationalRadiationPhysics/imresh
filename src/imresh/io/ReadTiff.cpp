/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Maximilian Knespel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


#include "ReadTiff.hpp"

#include <tiffio.h>
#include <cstdint>  // uint32_t
#include <iostream>
#include <iomanip>  // std::hex
#include <string>
#include <cmath>    // fmin
#include <cfloat>   // FLT_MIN, FLT_MAX
#include <cstring>  // memset
#include <utility>  // size_t
#include <cassert>
#include <cstddef>  // NULL
#include <map>


void ReadTiff::initializeInfo( void )
{
    auto & info = mTiffInfo;
    auto & av = mTiffInfoAvailable;

    memset( &info, 0, sizeof(info) );
    memset( &av, 0, sizeof(av) );

    av.artist                 = TIFFGetField( mImage, TIFFTAG_ARTIST                , &info.artist                 );
    av.badFaxLines            = TIFFGetField( mImage, TIFFTAG_BADFAXLINES           , &info.badFaxLines            );
    av.bitsPerSample          = TIFFGetField( mImage, TIFFTAG_BITSPERSAMPLE         , &info.bitsPerSample          );
    av.cleanFaxData           = TIFFGetField( mImage, TIFFTAG_CLEANFAXDATA          , &info.cleanFaxData           );
    av.colorMap               = TIFFGetField( mImage, TIFFTAG_COLORMAP              , &info.colorMap               );
    av.compression            = TIFFGetField( mImage, TIFFTAG_COMPRESSION           , &info.compression            );
    av.consecutiveBadFaxLines = TIFFGetField( mImage, TIFFTAG_CONSECUTIVEBADFAXLINES, &info.consecutiveBadFaxLines );
    av.copyright              = TIFFGetField( mImage, TIFFTAG_COPYRIGHT             , &info.copyright              );
    av.dataType               = TIFFGetField( mImage, TIFFTAG_DATATYPE              , &info.dataType               );
    av.dateTime               = TIFFGetField( mImage, TIFFTAG_DATETIME              , &info.dateTime               );
    av.documentName           = TIFFGetField( mImage, TIFFTAG_DOCUMENTNAME          , &info.documentName           );
    av.dotRange               = TIFFGetField( mImage, TIFFTAG_DOTRANGE              , &info.dotRange               );
    av.faxMode                = TIFFGetField( mImage, TIFFTAG_FAXMODE               , &info.faxMode                );
    av.fillOrder              = TIFFGetField( mImage, TIFFTAG_FILLORDER             , &info.fillOrder              );
    av.group3Options          = TIFFGetField( mImage, TIFFTAG_GROUP3OPTIONS         , &info.group3Options          );
    av.group4Options          = TIFFGetField( mImage, TIFFTAG_GROUP4OPTIONS         , &info.group4Options          );
    av.halfToneHints          = TIFFGetField( mImage, TIFFTAG_HALFTONEHINTS         , &info.halfToneHints          );
    av.hostComputer           = TIFFGetField( mImage, TIFFTAG_HOSTCOMPUTER          , &info.hostComputer           );
    av.imageDepth             = TIFFGetField( mImage, TIFFTAG_IMAGEDEPTH            , &info.imageDepth             );
    av.imageDescription       = TIFFGetField( mImage, TIFFTAG_IMAGEDESCRIPTION      , &info.imageDescription       );
    av.imageLength            = TIFFGetField( mImage, TIFFTAG_IMAGELENGTH           , &info.imageLength            );
    av.imageWidth             = TIFFGetField( mImage, TIFFTAG_IMAGEWIDTH            , &info.imageWidth             );
    av.inkNames               = TIFFGetField( mImage, TIFFTAG_INKNAMES              , &info.inkNames               );
    av.inkSet                 = TIFFGetField( mImage, TIFFTAG_INKSET                , &info.inkSet                 );
    av.jpegQuality            = TIFFGetField( mImage, TIFFTAG_JPEGQUALITY           , &info.jpegQuality            );
    av.jpegColorMode          = TIFFGetField( mImage, TIFFTAG_JPEGCOLORMODE         , &info.jpegColorMode          );
    av.jpegTablesMode         = TIFFGetField( mImage, TIFFTAG_JPEGTABLESMODE        , &info.jpegTablesMode         );
    av.make                   = TIFFGetField( mImage, TIFFTAG_MAKE                  , &info.make                   );
    av.matteing               = TIFFGetField( mImage, TIFFTAG_MATTEING              , &info.matteing               );
    av.maxSampleValue         = TIFFGetField( mImage, TIFFTAG_MAXSAMPLEVALUE        , &info.maxSampleValue         );
    av.minSampleValue         = TIFFGetField( mImage, TIFFTAG_MINSAMPLEVALUE        , &info.minSampleValue         );
    av.model                  = TIFFGetField( mImage, TIFFTAG_MODEL                 , &info.model                  );
    av.orientation            = TIFFGetField( mImage, TIFFTAG_ORIENTATION           , &info.orientation            );
    av.pageName               = TIFFGetField( mImage, TIFFTAG_PAGENAME              , &info.pageName               );
    av.pageNumber             = TIFFGetField( mImage, TIFFTAG_PAGENUMBER            , &info.pageNumber             );
    av.photometric            = TIFFGetField( mImage, TIFFTAG_PHOTOMETRIC           , &info.photometric            );
    av.planarConfig           = TIFFGetField( mImage, TIFFTAG_PLANARCONFIG          , &info.planarConfig           );
    av.predictor              = TIFFGetField( mImage, TIFFTAG_PREDICTOR             , &info.predictor              );
    av.resolutionUnit         = TIFFGetField( mImage, TIFFTAG_RESOLUTIONUNIT        , &info.resolutionUnit         );
    av.rowsPerStrip           = TIFFGetField( mImage, TIFFTAG_ROWSPERSTRIP          , &info.rowsPerStrip           );
    av.sampleFormat           = TIFFGetField( mImage, TIFFTAG_SAMPLEFORMAT          , &info.sampleFormat           );
    av.samplesPerPixel        = TIFFGetField( mImage, TIFFTAG_SAMPLESPERPIXEL       , &info.samplesPerPixel        );
    av.sMaxSampleValue        = TIFFGetField( mImage, TIFFTAG_SMAXSAMPLEVALUE       , &info.sMaxSampleValue        );
    av.sMinSampleValue        = TIFFGetField( mImage, TIFFTAG_SMINSAMPLEVALUE       , &info.sMinSampleValue        );
    av.software               = TIFFGetField( mImage, TIFFTAG_SOFTWARE              , &info.software               );
    av.subFileType            = TIFFGetField( mImage, TIFFTAG_SUBFILETYPE           , &info.subFileType            );
    av.targetPrinter          = TIFFGetField( mImage, TIFFTAG_TARGETPRINTER         , &info.targetPrinter          );
    av.thresholding           = TIFFGetField( mImage, TIFFTAG_THRESHHOLDING         , &info.thresholding           );
    av.tileDepth              = TIFFGetField( mImage, TIFFTAG_TILEDEPTH             , &info.tileDepth              );
    av.tileLength             = TIFFGetField( mImage, TIFFTAG_TILELENGTH            , &info.tileLength             );
    av.tileWidth              = TIFFGetField( mImage, TIFFTAG_TILEWIDTH             , &info.tileWidth              );
    av.transferFunction       = TIFFGetField( mImage, TIFFTAG_TRANSFERFUNCTION      , &info.transferFunction       );
}


void ReadTiff::readIntoBuffer( void )
{
    /* check size of image */
    unsigned int const nStrips = TIFFNumberOfStrips( mImage );
    unsigned int const stripSize = TIFFStripSize( mImage );
    size_t stripTotalSize = nStrips * stripSize;
    size_t imageTotalSize = mTiffInfo.imageWidth * mTiffInfo.imageLength
                          * ( mTiffInfo.bitsPerSample / 8 );
    assert( stripTotalSize == imageTotalSize );
    if ( stripTotalSize != imageTotalSize )
        throw StripTotalSizeNotMatchingImageDimensions;
    mnBytesBuffer = stripTotalSize;

    /* load image into buffer */
    switch( mTiffInfo.sampleFormat )
    {
        case SAMPLEFORMAT_IEEEFP:
        {
            assert( mBuffer == NULL && "Image was already read into buffer!" );
            mBuffer = new uint8_t[ mnBytesBuffer ];
            memset( mBuffer, 0, mnBytesBuffer );

            for ( auto iStrip = 0u; iStrip < nStrips; ++iStrip )
            {
                TIFFReadEncodedStrip( mImage, iStrip,
                    &( (uint8_t*) mBuffer )[ iStrip * stripSize ], tsize_t(-1) );
            }

            std::cout << "First values of image: ";
            for ( auto i = 0u; i < 16; ++i )
                std::cout << ( (float*) mBuffer )[i] << " ";
            std::cout << std::endl;
            std::cout << "Some values from the image center: ";
            for ( auto i = 0u; i < 16; ++i )
            {
                assert( i < getWidth() / 2 );
                std::cout << ( (float*) mBuffer )[ getHeight() / 2 * getWidth() + getWidth() / 2 + i ] << " ";
            }
            std::cout << std::endl;

            break;
        }
        default:
        {
            assert( false && "Unsupported TIFF format! (not floating point)" );
            break;
        }
    }
}

void * ReadTiff::unlinkInternalBuffer( void )
{
    auto const buffer = mBuffer;
    mBuffer = NULL;
    mnBytesBuffer = 0;
    return buffer;
}


ReadTiff::ReadTiff( std::string rFilename )
: mImage( NULL ), mBuffer( NULL ), mnBytesBuffer( 0 )
{
    /* disable warnings */
    mOldTiffErrorHandler = TIFFSetWarningHandler( NULL );

    /* open TIFF file */
    mImage = TIFFOpen( rFilename.c_str(), "r" );
    if ( not mImage )
        throw ErrorOnOpeningFile;

    mFirstDirectory = TIFFCurrentDirectory( mImage );
    std::cout << "Initial Directory is " << mFirstDirectory << std::endl;

    initializeInfo();
    if ( mTiffInfo.planarConfig != PLANARCONFIG_CONTIG )
        throw UnsupportedTiffType;
    //printInfo();

    readIntoBuffer();
}

ReadTiff::~ReadTiff()
{
    TIFFClose( mImage );
    TIFFSetWarningHandler( mOldTiffErrorHandler );

    if ( mBuffer != NULL )
    {
        delete[] mBuffer;
        mBuffer = NULL;
        mnBytesBuffer = 0;
    }
}

TIFF * ReadTiff::getRawTiff( void ) const
{
    return mImage;
}

unsigned int ReadTiff::getDirectoryCount( void )
{
    /* analyze if there are more than one image in this tiff file */
    auto iDirectory = TIFFCurrentDirectory( mImage );
    TIFFSetDirectory( mImage, mFirstDirectory );
    unsigned int nDirectories = 0;
    do
    {
        nDirectories++;
    }
    while ( TIFFReadDirectory( mImage ) );
    TIFFSetDirectory( mImage, iDirectory );

    std::cout << "TIFF file contains " << nDirectories
              << " images / directories" << std::endl;

    return nDirectories;
}

std::string ReadTiff::getCompressionString( uint16_t rCompression )
{
    switch ( rCompression )
    {
        case COMPRESSION_NONE         : return "none (dump mode)"                                ;
        case COMPRESSION_CCITTRLE     : return "CCITT modified Huffman RLE"                      ;
        //case COMPRESSION_CCITTFAX3    : return "CCITT Group 3 fax encoding"                      ;
        case COMPRESSION_CCITT_T4     : return "CCITT T.4 (TIFF 6 name)"                         ;
        //case COMPRESSION_CCITTFAX4    : return "CCITT Group 4 fax encoding"                      ;
        case COMPRESSION_CCITT_T6     : return "CCITT T.6 (TIFF 6 name)"                         ;
        case COMPRESSION_LZW          : return "Lempel-Ziv  & Welch"                             ;
        case COMPRESSION_OJPEG        : return "!6.0 JPEG"                                       ;
        case COMPRESSION_JPEG         : return "%JPEG DCT compression"                           ;
        case COMPRESSION_T85          : return "!TIFF/FX T.85 JBIG compression"                  ;
        case COMPRESSION_T43          : return "!TIFF/FX T.43 colour by layered JBIG compression";
        case COMPRESSION_NEXT         : return "NeXT 2-bit RLE"                                  ;
        case COMPRESSION_CCITTRLEW    : return "#1 w/ word alignment"                            ;
        case COMPRESSION_PACKBITS     : return "Macintosh RLE"                                   ;
        case COMPRESSION_THUNDERSCAN  : return "ThunderScan RLE"                                 ;
        case COMPRESSION_IT8CTPAD     : return "IT8 CT w/padding"                                ;
        case COMPRESSION_IT8LW        : return "IT8 Linework RLE"                                ;
        case COMPRESSION_IT8MP        : return "IT8 Monochrome picture"                          ;
        case COMPRESSION_IT8BL        : return "IT8 Binary line art"                             ;
        case COMPRESSION_PIXARFILM    : return "Pixar companded 10bit LZW"                       ;
        case COMPRESSION_PIXARLOG     : return "Pixar companded 11bit ZIP"                       ;
        case COMPRESSION_DEFLATE      : return "Deflate compression"                             ;
        case COMPRESSION_ADOBE_DEFLATE: return "Deflate compression, as recognized by Adobe"     ;
        case COMPRESSION_DCS          : return "Kodak DCS encoding"                              ;
        case COMPRESSION_JBIG         : return "ISO JBIG"                                        ;
        case COMPRESSION_SGILOG       : return "SGI Log Luminance RLE"                           ;
        case COMPRESSION_SGILOG24     : return "SGI Log 24-bit packed"                           ;
        case COMPRESSION_JP2000       : return "Leadtools JPEG2000"                              ;
        case COMPRESSION_LZMA         : return "LZMA2"                                           ;
        default                       : return "unknown compression";
        /* codes 32895-32898 are reserved for ANSI IT8 TIFF/IT <dkelly@apago.com) */
        /* compression codes 32908-32911 are reserved for Pixar */
        /* compression code 32947 is reserved for Oceana Matrix <dev@oceana.com> */
    }
}

std::string ReadTiff::getPhotometricInterpretationString( uint16_t rPhotometric )
{
    switch ( rPhotometric )
    {
        case PHOTOMETRIC_MINISWHITE: return "min value is white";
        case PHOTOMETRIC_MINISBLACK: return "min value is black";
        case PHOTOMETRIC_RGB       : return "RGB color model";
        case PHOTOMETRIC_PALETTE   : return "color map indexed";
        case PHOTOMETRIC_MASK      : return "$holdout mask";
        case PHOTOMETRIC_SEPARATED : return "!color separations";
        case PHOTOMETRIC_YCBCR     : return "!CCIR 601";
        case PHOTOMETRIC_CIELAB    : return "!1976 CIE L*a*b*";
        case PHOTOMETRIC_ICCLAB    : return "ICC L*a*b* [Adobe TIFF Technote 4]";
        case PHOTOMETRIC_ITULAB    : return "ITU L*a*b*";
        case PHOTOMETRIC_CFA       : return "color filter array";
        case PHOTOMETRIC_LOGL      : return "CIE Log2(L)";
        case PHOTOMETRIC_LOGLUV    : return "CIE Log2(L) (u',v')";
        default                    : return "unknown photmetric interpretation";
    }
}

std::string ReadTiff::getPlanarConfigString( uint16_t planarConfig )
{
    switch ( planarConfig )
    {
        case PLANARCONFIG_CONTIG  : return "single image plane";
        case PLANARCONFIG_SEPARATE: return "separate planes of data";
        default                   : return "unknown configuration";
    }
}

std::string ReadTiff::getSampleFormatString( uint16_t sampleFormat )
{
    switch ( sampleFormat )
    {
        case SAMPLEFORMAT_UINT         : return "unsigned integer data";
        case SAMPLEFORMAT_INT          : return "signed integer data";
        case SAMPLEFORMAT_IEEEFP       : return "IEEE floating point data";
        case SAMPLEFORMAT_VOID         : return "untyped data";
        case SAMPLEFORMAT_COMPLEXINT   : return "complex signed int";
        case SAMPLEFORMAT_COMPLEXIEEEFP: return "complex ieee floating";
        default                        : return "unknown sample format";
    }
}

void ReadTiff::printInfo( void ) const
{
    if ( TIFFIsTiled( mImage ) )
    {
        std::cout << "Image is tiled!" << std::endl;
    }
    if ( TIFFIsByteSwapped( mImage ) )
    {
        std::cout << "Image has different byte-order than machine." << std::endl;
    }
    if ( TIFFIsMSB2LSB( mImage ) )
    {
        std::cout << "Bit 0 is most significant bit!" << std::endl;
    }

    std::cout << "TIFFStripSize =      " << TIFFStripSize     ( mImage ) << std::endl;
    std::cout << "TIFFNumberOfStrips = " << TIFFNumberOfStrips( mImage ) << std::endl;

    auto & info = mTiffInfo;
    auto & av   = mTiffInfoAvailable;

    if ( av.artist                 ) std::cout << "artist                 = " << info.artist                 << "\n";
    if ( av.badFaxLines            ) std::cout << "badFaxLines            = " << info.badFaxLines            << "\n";
    if ( av.bitsPerSample          ) std::cout << "bitsPerSample          = " << info.bitsPerSample          << "\n";
    if ( av.cleanFaxData           ) std::cout << "cleanFaxData           = " << info.cleanFaxData           << "\n";
    if ( av.colorMap               ) std::cout << "colorMap[0]            = " << info.colorMap[0]            << "\n";
    if ( av.colorMap               ) std::cout << "colorMap[1]            = " << info.colorMap[1]            << "\n";
    if ( av.colorMap               ) std::cout << "colorMap[2]            = " << info.colorMap[2]            << "\n";
    if ( av.consecutiveBadFaxLines ) std::cout << "consecutiveBadFaxLines = " << info.consecutiveBadFaxLines << "\n";
    if ( av.copyright              ) std::cout << "copyright              = " << info.copyright              << "\n";
    if ( av.dataType               ) std::cout << "dataType               = " << info.dataType               << "\n";
    if ( av.dateTime               ) std::cout << "dateTime               = " << info.dateTime               << "\n";
    if ( av.documentName           ) std::cout << "documentName           = " << info.documentName           << "\n";
    if ( av.dotRange               ) std::cout << "dotRange[0]            = " << info.dotRange[0]            << "\n";
    if ( av.dotRange               ) std::cout << "dotRange[1]            = " << info.dotRange[1]            << "\n";
    if ( av.faxMode                ) std::cout << "faxMode                = " << info.faxMode                << "\n";
    if ( av.fillOrder              ) std::cout << "fillOrder              = " << info.fillOrder              << "\n";
    if ( av.group3Options          ) std::cout << "group3Options          = " << info.group3Options          << "\n";
    if ( av.group4Options          ) std::cout << "group4Options          = " << info.group4Options          << "\n";
    if ( av.halfToneHints          ) std::cout << "halfToneHints[0]       = " << info.halfToneHints[0]       << "\n";
    if ( av.halfToneHints          ) std::cout << "halfToneHints[1]       = " << info.halfToneHints[1]       << "\n";
    if ( av.hostComputer           ) std::cout << "hostComputer           = " << info.hostComputer           << "\n";
    if ( av.imageDepth             ) std::cout << "imageDepth             = " << info.imageDepth             << "\n";
    if ( av.imageDescription       ) std::cout << "imageDescription       = " << info.imageDescription       << "\n";
    if ( av.imageLength            ) std::cout << "imageLength            = " << info.imageLength            << "\n";
    if ( av.imageWidth             ) std::cout << "imageWidth             = " << info.imageWidth             << "\n";
    if ( av.inkNames               ) std::cout << "inkNames               = " << info.inkNames               << "\n";
    if ( av.inkSet                 ) std::cout << "inkSet                 = " << info.inkSet                 << "\n";
    if ( av.jpegQuality            ) std::cout << "jpegQuality            = " << info.jpegQuality            << "\n";
    if ( av.jpegColorMode          ) std::cout << "jpegColorMode          = " << info.jpegColorMode          << "\n";
    if ( av.jpegTablesMode         ) std::cout << "jpegTablesMode         = " << info.jpegTablesMode         << "\n";
    if ( av.make                   ) std::cout << "make                   = " << info.make                   << "\n";
    if ( av.matteing               ) std::cout << "matteing               = " << info.matteing               << "\n";
    if ( av.maxSampleValue         ) std::cout << "maxSampleValue         = " << info.maxSampleValue         << "\n";
    if ( av.minSampleValue         ) std::cout << "minSampleValue         = " << info.minSampleValue         << "\n";
    if ( av.model                  ) std::cout << "model                  = " << info.model                  << "\n";
    if ( av.orientation            ) std::cout << "orientation            = " << info.orientation            << "\n";
    if ( av.pageName               ) std::cout << "pageName               = " << info.pageName               << "\n";
    if ( av.pageNumber             ) std::cout << "pageNumber             = " << info.pageNumber             << "\n";
    if ( av.predictor              ) std::cout << "predictor              = " << info.predictor              << "\n";
    if ( av.resolutionUnit         ) std::cout << "resolutionUnit         = " << info.resolutionUnit         << "\n";
    if ( av.rowsPerStrip           ) std::cout << "rowsPerStrip           = " << info.rowsPerStrip           << "\n";
    if ( av.samplesPerPixel        ) std::cout << "samplesPerPixel        = " << info.samplesPerPixel        << "\n";
    if ( av.software               ) std::cout << "software               = " << info.software               << "\n";
    if ( av.subFileType            ) std::cout << "subFileType            = " << info.subFileType            << "\n";
    if ( av.targetPrinter          ) std::cout << "targetPrinter          = " << info.targetPrinter          << "\n";
    if ( av.thresholding           ) std::cout << "thresholding           = " << info.thresholding           << "\n";
    if ( av.tileDepth              ) std::cout << "tileDepth              = " << info.tileDepth              << "\n";
    if ( av.tileLength             ) std::cout << "tileLength             = " << info.tileLength             << "\n";
    if ( av.tileWidth              ) std::cout << "tileWidth              = " << info.tileWidth              << "\n";
    if ( av.transferFunction       ) std::cout << "transferFunction[0]    = " << info.transferFunction[0]    << "\n";
    if ( av.transferFunction       ) std::cout << "transferFunction[1]    = " << info.transferFunction[1]    << "\n";
    if ( av.transferFunction       ) std::cout << "transferFunction[2]    = " << info.transferFunction[2]    << "\n";

    std::cout
    << "compression            = " << getCompressionString( info.compression ) << "\n"
    << "photometric            = " << getPhotometricInterpretationString( info.photometric ) << "\n"
    << "sMinSampleValue        = " << info.sMinSampleValue << "( FLT_MIN = " << FLT_MIN << ", DBL_MIN = " << DBL_MIN << " )\n"
    << "sMaxSampleValue        = " << info.sMaxSampleValue << "( FLT_MAX = " << FLT_MAX << ", DBL_MAX = " << DBL_MAX << " )\n"
    << "planarConfig           = " << getPlanarConfigString( info.planarConfig ) << "\n"
    << "sampleFormat           = " << getSampleFormatString( info.sampleFormat ) << "\n"
    << std::flush;
}

auto ReadTiff::getWidth( void ) const
-> decltype( mTiffInfo.imageWidth )
{
    return mTiffInfo.imageWidth;
}

auto ReadTiff::getHeight( void ) const
-> decltype( mTiffInfo.imageLength )
{
    return mTiffInfo.imageLength;
}

auto ReadTiff::getSampleFormat( void ) const
 -> decltype( mTiffInfo.sampleFormat )
{
    return mTiffInfo.sampleFormat;
}

auto ReadTiff::getBufferSize( void ) const
 -> decltype( mnBytesBuffer )
{
    return mnBytesBuffer;
}

float ReadTiff::getPixel
(
    unsigned int const iX,
    unsigned int const iY
) const
{
    assert( iX < getWidth () );
    assert( iY < getHeight() );

    switch( mTiffInfo.sampleFormat )
    {
        case SAMPLEFORMAT_IEEEFP:
        {
            assert( mBuffer != NULL );
            auto buffer = (float*) mBuffer;
            return buffer[ iY * getWidth() + iX ];
        }
        default:
        {
            assert( false && "Unsupported TIFF format! (not floating point)" );
            return -1;
        }
    }
}

ReadTiff::ColorInfo ReadTiff::getColorInfo( void ) const
{
    ColorInfo results;
    results.min   =  FLT_MAX;
    results.max   = -FLT_MAX;
    results.mean  =  0;
    results.count =  0;

    using UniqueColors = std::map< float, int >;
    using KeyValue = UniqueColors::value_type;
    UniqueColors colors;

    auto const Nx = getWidth ();
    auto const Ny = getHeight();
    for ( auto ix = 0u; ix < Nx; ++ix )
    {
        for ( auto iy = 0u; iy < Ny; ++iy )
        {
            auto const color = getPixel( ix, iy );
            auto const position = colors.insert( KeyValue{ color, 1 } );
            if ( position.second == false /* key already existed */ )
            {
                position.first->second += 1;
            }
            else /* new color found */
            {
                results.min = std::fmin( results.min, color );
                results.max = std::fmax( results.max, color );
            }
            results.mean += color;
        }
    }

    results.count = colors.size();
    results.mean /= results.count;

    std::cout                                << "\n"
    << "Color Information:"                  << "\n"
    << "    min   value = " << results.min   << "\n"
    << "    max   value = " << results.max   << "\n"
    << "    mean  value = " << results.mean  << "\n"
    << "    count value = " << results.count << "\n"
    << std::endl;

    return results;
}
