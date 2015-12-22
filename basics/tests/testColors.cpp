
#include "testColors.h"


namespace imresh {
namespace test {


void testHsv( SDL_Renderer * rpRenderer )
{
    using namespace sdlcommon;
    using namespace imresh::colors;

    SDL_Rect rect = { 40,40, 100, 100 };
    for ( float saturation = 0.4; saturation <= 1.0; saturation += 0.2 )
    {
        SDL_SetRenderDrawColor( rpRenderer, 0,0,0,1 );
        int w = 3;
        SDL_Rect boundingBox = SDL_Rect{ rect.x-w, rect.y-w, rect.w+2*w, rect.h+2*w };
        SDL_RenderDrawThickRect( rpRenderer, &boundingBox, w );

        for ( int ix = 0; ix < rect.w; ++ix )
        for ( int iy = 0; iy < rect.h; ++iy )
        {
            float r,g,b;
            hsvToRgb( (float)ix/rect.w * 2*M_PI, saturation, (float)iy/rect.h,
                      &r,&g,&b );
            SDL_SetRenderDrawColor( rpRenderer, 255*r,255*g,255*b, /*a*/ 1 );
            SDL_RenderDrawPoint( rpRenderer, rect.x+ix, rect.y+(rect.h-1)-iy );
        }
        rect.x += 40 + rect.w;
    }

    /* check correct working of RGB->HSV with X11 color table */
    for ( const auto & color : x11Colors )
    {
        float r,g,b,error;
        auto rgb = color.second[ 0 /*RGB*/ ];
        auto hsv = color.second[ 2 /*HSV*/ ];
        hsvToRgb( hsv[0], hsv[1], hsv[2], &r, &g, &b );
        error = fmax( fmax( fabs( rgb[0]-r ),
                            fabs( rgb[1]-g ) ),
                      fabs( rgb[2]-b ) );
        if ( error > 0.01 )
        {
            std::cout << "HSV Deviation for '"<<color.first<<"'="<<error<<" is too large!\n"
            << "    HSV real: ("<<hsv[0]<<","<<hsv[1]<<","<<hsv[2]<<")\n"
            << "    RGB real: ("<<rgb[0]<<","<<rgb[1]<<","<<rgb[2]<<")\n"
            << "    RGB calc: ("<<r<<","<<g<<","<<b<<")\n";
        }
        assert( error <= 0.01 );
    }
}

void testHsl( SDL_Renderer * rpRenderer )
{
    using namespace sdlcommon;
    using namespace imresh::colors;

    SDL_Rect rect = { 40,180, 100, 100 };
    for ( float saturation = 0.4; saturation <= 1.0; saturation += 0.2 )
    {
        SDL_SetRenderDrawColor( rpRenderer, 0,0,0,1 );
        int w = 3;
        SDL_Rect boundingBox = SDL_Rect{ rect.x-w, rect.y-w, rect.w+2*w, rect.h+2*w };
        SDL_RenderDrawThickRect( rpRenderer, &boundingBox, w );

        for ( int ix = 0; ix < rect.w; ++ix )
        for ( int iy = 0; iy < rect.h; ++iy )
        {
            float r,g,b;
            hslToRgb( (float)ix/rect.w * 2*M_PI, saturation, (float)iy/rect.h,
                      &r,&g,&b );
            SDL_SetRenderDrawColor( rpRenderer, 255*r,255*g,255*b, /*a*/ 1 );
            SDL_RenderDrawPoint( rpRenderer, rect.x+ix, rect.y+rect.h-iy );
        }
        rect.x += 40 + rect.w;
    }

    /* check correct working of RGB->HSV with X11 color table */
    for ( const auto & color : x11Colors )
    {
        float r,g,b;
        auto rgb = color.second[ 0 /*RGB*/ ];
        auto hsl = color.second[ 1 /*HSL*/ ];
        hslToRgb( hsl[0], hsl[1], hsl[2], &r, &g, &b );
        const float error = fmax( fmax( fabs( rgb[0]-r ),
                                        fabs( rgb[1]-g ) ),
                                  fabs( rgb[2]-b ) );
        const float maxErr = 0.015;
        /* don't use error > maxErr, because it is wrong for NaN ... */
        if ( not ( error <= maxErr ) )
        {
            std::cout << "HSL Deviation for '"<<color.first<<"' = "<<error<<" is too large!\n"
            << "    HSV real: ("<<hsl[0]<<","<<hsl[1]<<","<<hsl[2]<<")\n"
            << "    RGB real: ("<<rgb[0]<<","<<rgb[1]<<","<<rgb[2]<<")\n"
            << "    RGB calc: ("<<r<<","<<g<<","<<b<<")\n" << std::flush;
        }
        assert( error <= maxErr );
    }
}


} // namespace imresh
} // namespace test
