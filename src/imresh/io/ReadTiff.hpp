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


#pragma once


#include <tiffio.h> // TIFF *, TIFFErrorHandler
#include <cstdint>  // uint8_t, uint32_t, ...
#include <string>
#include <utility>  // size_t


class ReadTiff
{
    static int constexpr nMaxChars = 1024; // for char-arrays
    using size_t = std::size_t;

public:

    enum ErrorCodes {
        ErrorOnOpeningFile,
        UnsupportedTiffType,
        StripTotalSizeNotMatchingImageDimensions,
    };

    struct TiffInfo
    {
        char             artist                [nMaxChars];
        uint32_t         badFaxLines                      ;
        uint16_t         bitsPerSample                    ;
        uint16_t         cleanFaxData                     ;
        uint16_t         colorMap              [3]        ;
        uint16_t         compression                      ;
        uint32_t         consecutiveBadFaxLines           ;
        char             copyright             [nMaxChars];
        uint16_t         dataType                         ;
        char             dateTime              [nMaxChars];
        char             documentName          [nMaxChars];
        uint16_t         dotRange              [2]        ;
        int              faxMode                          ;
        uint16_t         fillOrder                        ;
        uint32_t         group3Options                    ;
        uint32_t         group4Options                    ;
        uint16_t         halfToneHints         [2]        ;
        char             hostComputer          [nMaxChars];
        uint32_t         imageDepth                       ;
        char             imageDescription      [nMaxChars];
        uint32_t         imageLength                      ;
        uint32_t         imageWidth                       ;
        char             inkNames              [nMaxChars];
        uint16_t         inkSet                           ;
        int              jpegQuality                      ;
        int              jpegColorMode                    ;
        int              jpegTablesMode                   ;
        char             make                  [nMaxChars];
        uint16_t         matteing                         ;
        uint16_t         maxSampleValue                   ;
        uint16_t         minSampleValue                   ;
        char             model                 [nMaxChars];
        uint16_t         orientation                      ;
        char             pageName              [nMaxChars];
        uint16_t         pageNumber                       ;
        uint16_t         photometric                      ;
        uint16_t         planarConfig                     ;
        uint16_t         predictor                        ;
        uint16_t         resolutionUnit                   ;
        uint32_t         rowsPerStrip                     ;
        uint16_t         sampleFormat                     ;
        uint16_t         samplesPerPixel                  ;
        double           sMaxSampleValue                  ;
        double           sMinSampleValue                  ;
        char             software              [nMaxChars];
        uint32           subFileType                      ;
        char             targetPrinter         [nMaxChars];
        uint16           thresholding                     ;
        uint32           tileDepth                        ;
        uint32           tileLength                       ;
        uint32           tileWidth                        ;
        uint16           transferFunction      [3]        ;

        //float**           whitepoint
        //float*            xposition
        //float*            xresolution
        //float**           ycbcrcoefficients
        //uint16*           ycbcrpositioning
        //uint16*         2      ycbcrsubsampling
        //float*          1      yposition
        //float*          1      yresolution
        //uint32*,void**  2      iccprofile

        //uint16*,uint16**2      extrasamples
        //TIFFFaxFillFunc*       faxfillfunc
        //u_short*,void** 2      jpegtables
        //uint16*,uint32**2      subifd
        //float*           primarychromaticities
        //float*           referenceblackwhite
        //double*          stonits
        //uint32*          stripbytecounts
        //uint32*          stripoffsets
        //uint32*          tilebytecounts
        //uint32*          tileoffsets
    };

    struct TiffInfoAvailable
    {
        bool artist                ;
        bool badFaxLines           ;
        bool bitsPerSample         ;
        bool cleanFaxData          ;
        bool colorMap              ;
        bool compression           ;
        bool consecutiveBadFaxLines;
        bool copyright             ;
        bool dataType              ;
        bool dateTime              ;
        bool documentName          ;
        bool dotRange              ;
        bool faxMode               ;
        bool fillOrder             ;
        bool group3Options         ;
        bool group4Options         ;
        bool halfToneHints         ;
        bool hostComputer          ;
        bool imageDepth            ;
        bool imageDescription      ;
        bool imageLength           ;
        bool imageWidth            ;
        bool inkNames              ;
        bool inkSet                ;
        bool jpegQuality           ;
        bool jpegColorMode         ;
        bool jpegTablesMode        ;
        bool make                  ;
        bool matteing              ;
        bool maxSampleValue        ;
        bool minSampleValue        ;
        bool model                 ;
        bool orientation           ;
        bool pageName              ;
        bool pageNumber            ;
        bool photometric           ;
        bool planarConfig          ;
        bool predictor             ;
        bool resolutionUnit        ;
        bool rowsPerStrip          ;
        bool sampleFormat          ;
        bool samplesPerPixel       ;
        bool sMaxSampleValue       ;
        bool sMinSampleValue       ;
        bool software              ;
        bool subFileType           ;
        bool targetPrinter         ;
        bool thresholding          ;
        bool tileDepth             ;
        bool tileLength            ;
        bool tileWidth             ;
        bool transferFunction      ;
    };

private:

    TIFF * mImage;
    TIFFErrorHandler mOldTiffErrorHandler;
    uint16_t mFirstDirectory;
    TiffInfo mTiffInfo;
    TiffInfoAvailable mTiffInfoAvailable;
    uint8_t * mBuffer;
    size_t mnBytesBuffer;

    void initializeInfo( void );

public:

    void readIntoBuffer( void );
    void * unlinkInternalBuffer( void );

    ReadTiff();
    ReadTiff( std::string rFilename );
    ~ReadTiff();

    TIFF * getRawTiff( void ) const;
    unsigned int getDirectoryCount( void );

    static std::string getCompressionString( uint16_t rCompression );
    static std::string getPhotometricInterpretationString( uint16_t rPhotometric );
    static std::string getPlanarConfigString( uint16_t planarConfig );
    static std::string getSampleFormatString( uint16_t sampleFormat );

    void printInfo( void ) const;

    auto getWidth       ( void ) const -> decltype( mTiffInfo.imageWidth   );
    auto getHeight      ( void ) const -> decltype( mTiffInfo.imageLength  );
    auto getSampleFormat( void ) const -> decltype( mTiffInfo.sampleFormat );
    auto getBufferSize  ( void ) const -> decltype( mnBytesBuffer          );

    float getPixel
    (
        unsigned int const iX,
        unsigned int const iY
    ) const;

    /* double would be better, as it would allow perfect downwards
     * compatibility to 32 bit integers, but e.g. on CUDA double may
     * take much more time to calculate */
    struct ColorInfo
    {
        float min  ;
        float max  ;
        float mean ;
        float count;
    };

    ColorInfo getColorInfo( void ) const;
};

