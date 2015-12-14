#include "dcft.h"

template<class F, class G>
float trigScp(const F & f, const G & g)
{
    const float a  = -M_PI;
    const float b  =  M_PI;
    const float N  = 1e5;
    const float dx = (b-a) / N;

    struct Integrand {
        const F & f; const G & g;
        int N; float a; float dx;
        int size(void) const { return N; }
        float operator[]( int i ) const { return f(a+i*dx)*g(a+i*dx); }
    } integrand({ f,g,(int)N,a,dx });

    return numeric::integrate::trapezoid( integrand, dx ) / M_PI;
}

template<class F>
void dcft( F f, int rnCoefficients, float * rCoefficients )
{
    for ( int i = 0; i < rnCoefficients; ++i )
        rCoefficients[i] = trigScp( f, CosBase({i}) ) / M_PI;
    for ( int i = 0; i < rnCoefficients; ++i )
        rCoefficients[rnCoefficients+i] = trigScp( f, SinBase({i}) ) / M_PI;
}


template<class F>
void testDcftAndPrint
( SDL_Renderer * rpRenderer, SDL_Rect rAxes, F f, const char* rFuncTitle )
{
    int nCoefficients = 10;
    float * g = new float[2*nCoefficients];
    dcft( f,nCoefficients,g );

    std::cout << "Coefficients for " << rFuncTitle << ":\n  ";
    for ( int i = 0; i < 2*nCoefficients; ++i )
        std::cout << g[i] << " ";
    std::cout << "\n\n";

    /******************* plot original function *******************/
    SDL_RenderDrawFunction( rpRenderer, rAxes, -M_PI,M_PI ,0,0, f, /*drawAxis*/ true );
    SDL_RenderDrawArrow( rpRenderer, rAxes.x+rAxes.w*1.2, rAxes.y+rAxes.h/2,
                                     rAxes.x+rAxes.w*1.4, rAxes.y+rAxes.h/2 );

    /******************* plot coefficients *******************/
    char title[128];
    sprintf(title,"Re FT[%s]",rFuncTitle);
    rAxes.x += 2.0*rAxes.w;
    SDL_RenderDrawHistogram(rpRenderer, rAxes, 0,nCoefficients-1, 0,0,
        g,nCoefficients, 0,false/*fill*/, true/*drawAxis*/, title );

    sprintf(title,"Im FT[%s]",rFuncTitle);
    rAxes.x += 1.5*rAxes.w;
    SDL_RenderDrawHistogram(rpRenderer, rAxes, 0,nCoefficients-1, 0,0,
        g+nCoefficients,nCoefficients, 0,false/*fill*/, true/*drawAxis*/, title );

    /******************* plot reconstructed function *******************/
    struct {
        float * coefficients;
        int nCoefficients;
        float operator()( float x )
        {
            float sum = 0;
            for ( int k = 0; k < nCoefficients; ++k )
                sum += coefficients[k] * cos( k*x );
            for ( int k = 0; k < nCoefficients; ++k )
                sum += coefficients[nCoefficients+k] * sin( k*x );
            return sum;
        }
    } fRec{ g,nCoefficients };
    SDL_RenderDrawArrow( rpRenderer, rAxes.x+rAxes.w*1.2, rAxes.y+rAxes.h/2,
                                     rAxes.x+rAxes.w*1.4, rAxes.y+rAxes.h/2 );
    rAxes.x += 2.0*rAxes.w;
    sprintf(title,"IFT[FT[%s]]",rFuncTitle);
    SDL_RenderDrawFunction( rpRenderer, rAxes, -M_PI,M_PI ,0,0, fRec, /*drawAxis*/ true );

    delete[] g;
}

void testDcft(SDL_Renderer * rpRenderer)
{
    SDL_Rect axes = { 40,40, 100,80 };
    testDcftAndPrint( rpRenderer,axes, [](float x){return std::sin(1*x);}, "Sin(x)"  ); axes.y += 110;
    testDcftAndPrint( rpRenderer,axes, [](float x){return std::sin(2*x);}, "Sin(2x)" ); axes.y += 110;
    testDcftAndPrint( rpRenderer,axes, [](float x){return std::sin(3*x);}, "Sin(3x)" ); axes.y += 110;
    testDcftAndPrint( rpRenderer,axes, [](float x){return std::abs(x);  }, "|x|"     ); axes.y += 110;
    testDcftAndPrint( rpRenderer,axes, [](float x){return x>0?1:0;      }, "Theta(x)" );
}
