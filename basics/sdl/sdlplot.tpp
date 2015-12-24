
namespace sdlcommon {


template<class T_FUNC>
int SDL_RenderDrawFunction
( SDL_Renderer * rpRenderer, const SDL_Rect & rAxes,
  float x0, float x1, float y0, float y1, T_FUNC f, bool drawAxis )
{
    const int &x=rAxes.x, &y=rAxes.y, &w=rAxes.w, &h=rAxes.h;

    /* find minimum and maximum function value for plot range */
    if ( y0 == y1 )
    {
        y0 = FLT_MAX;
        y1 = FLT_MIN;
        for ( float xValue = x0; xValue <= x1; xValue += (x1-x0)/100. )
        {
            y0 = fmin( y0, (float) f(xValue) );
            y1 = fmax( y1, (float) f(xValue) );
        }
        const float ySpan = y1-y0;
        y0 -= 0.1*ySpan;
        y1 += 0.1*ySpan;
    }
    if (x0 > x1) return 1;
    if (y0 > y1) return 1;

    /* go pixel for pixel on x-Axis, evaluate and plot corresponding value */
    float ypx0;
    float ypx1 = y+h - ( f(x0)-y0 )/( y1-y0 )*h;
    for ( int ix = x; ix < x+w; ++ix )
    {
        /* rotate function evaluations */
        ypx0 = ypx1;

        /* convert pixel x to function argument x by shifting and scaling */
        float xval = x0 + (ix-x + 0.5f)/w * (x1-x0);
        /* convert function value back to pixel by shifting and scaling */
        ypx1 = y+h - ( f(xval)-y0 )/( y1-y0 )*h;

        SDL_RenderDrawLine( rpRenderer, ix,ypx0, ix,ypx1 );
        /* @see https://bugzilla.libsdl.org/show_bug.cgi?id=3182 */
        SDL_RenderDrawPoint( rpRenderer, ix,ypx0 );
        SDL_RenderDrawPoint( rpRenderer, ix,ypx1 );
    }

    if ( drawAxis )
        SDL_RenderDrawAxes(rpRenderer,rAxes,x0,x1,y0,y1);

    return 0;
}


} // namespace sdlcommon
