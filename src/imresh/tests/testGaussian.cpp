/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2016 Maximilian Knespel
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


#include "testGaussian.h"


namespace imresh
{
namespace test
{


    void testGaussianBlurVector
    (
        SDL_Renderer * const & rpRenderer,
        SDL_Rect rRect,
        float * const & rData,
        const unsigned & rnData,
        const float & rSigma,
        const char * const & rTitle
    )
    {
        using namespace sdlcommon;
        using namespace imresh::algorithms;

        SDL_RenderDrawHistogram( rpRenderer, rRect, 0,0,0,0,
            rData,rnData, 0/*binWidth*/,false/*fill*/, true/*drawAxis*/, rTitle );
        SDL_RenderDrawArrow( rpRenderer,
                             rRect.x + 1.1*rRect.w, rRect.y + rRect.h/2,
                             rRect.x + 1.3*rRect.w, rRect.y + rRect.h/2 );
        rRect.x += 1.5*rRect.w;

        gaussianBlur( rData, rnData, rSigma );

        char title2[128];
        sprintf( title2, "G(s=%0.1f)*%s", rSigma, rTitle );
        SDL_RenderDrawHistogram( rpRenderer, rRect, 0,0,0,0, rData, rnData,
            0 /*binWidth*/, false /*fill*/, true /*drawAxis*/, title2 );
    }


    void testGaussian
    (
        SDL_Renderer * const & rpRenderer
    )
    {
        srand(165158631);
        SDL_Rect rect = { 40,40,200,80 };

        const int nData = 50;
        float data[nData];

        /* Try different data sets */
        for ( int i = 0; i < nData; ++i )
            data[i] = 255*rand()/(double)RAND_MAX;
        testGaussianBlurVector( rpRenderer,rect,data,nData, 1.0, "Random" );
        rect.y += 100;
        for ( int i = 0; i < nData; ++i )
            data[i] = 255*rand()/(double)RAND_MAX;
        testGaussianBlurVector( rpRenderer,rect,data,nData, 2.0, "Random" );
        rect.y += 100;
        for ( int i = 0; i < nData; ++i )
            data[i] = 255*rand()/(double)RAND_MAX;
        testGaussianBlurVector( rpRenderer,rect,data,nData, 4.0, "Random" );
        rect.y += 100;

        for ( int i = 0; i < nData; ++i )
            data[i] = i > nData/2 ? 1 : 0;
        testGaussianBlurVector( rpRenderer,rect,data,nData, 1.0, "Step" );
        rect.y += 100;
        for ( int i = 0; i < nData; ++i )
            data[i] = i > nData/2 ? 1 : 0;
        testGaussianBlurVector( rpRenderer,rect,data,nData, 4.0, "Step" );
        rect.y += 100;

        {
        const int nData2 = 100;
        float data2[nData2];
        float sigma = 8.0;
        float a =  1.0/( sqrt(2.0*M_PI)*sigma );
        float b = -1.0/( 2.0*sigma*sigma );
        for ( int i = 0; i < nData2; ++i )
            data2[i] = a*exp( (i-nData2/2)*(i-nData2/2)*b );
        char title[64];
        sprintf(title,"G(s=%.2f)",sigma);

        /* a guassian with @f[ \mu_1, \sigma_1 @f] convoluted with a gaussian
         * with @f[ \mu_1, \sigma_1 @f] results also in a gaussian with
         * @f[ \mu = \mu_1+\mu_2, \sigma = \sqrt{ \sigma_1^2+\sigma_2^2 } @f] */
        testGaussianBlurVector( rpRenderer,rect,data2,nData2, sigma, title );
        rect.y += 100;
        }
    }


} // namespace imresh
} // namespace test
