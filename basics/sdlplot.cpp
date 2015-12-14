
int SDL_drawString
( SDL_Renderer * rpRenderer, TTF_Font * rFont, const char * rString,
  SDL_Rect * rTarget, int rXAlign, int rYAlign )
{
    /* Ignore empty strings */
    if ( rString[0] == 0 )
        return 1;

    SDL_Color fc; /* use same font color as rendering color */
    SDL_GetRenderDrawColor( rpRenderer, &fc.r, &fc.g, &fc.b, &fc.a );

    SDL_Surface * sString = TTF_RenderText_Blended( rFont, rString, fc );
    CHECK_TTF_PTR( sString );
    SDL_Texture * tString = SDL_CreateTextureFromSurface( rpRenderer, sString );
    SDL_CHECK( tString );

    if ( rTarget->w <= 0 )
        rTarget->w = sString->w;
    if ( rTarget->h <= 0 )
        rTarget->h = sString->h;

    if ( rXAlign == 1 )
        rTarget->x -= rTarget->w/2;
    else if ( rXAlign == 2 )
        rTarget->x -= rTarget->w;

    if ( rYAlign == 1 )
        rTarget->y -= rTarget->h/2;
    else if ( rYAlign == 2 )
        rTarget->y -= rTarget->h;

    SDL_RenderCopy( rpRenderer, tString, NULL, rTarget );

    if ( tString != NULL ) SDL_DestroyTexture( tString );
    if ( sString != NULL ) SDL_FreeSurface   ( sString );

    return 0;
}

float chooseTickSpacing(float a, float b)
{
    float dx = b-a;
    /* if dx is 0.2, then 0.1 may be a good spacing, meaning search for
     * 1eX which is smaller than dx */
    float spacing = pow(10, floor( log10(dx)+0.5 )-1.0 );
    /* Problematic cases are the border cases like 0.1, 10, ..., because
     * no rounding occurs, meaning we also get a spacing of 0.1, resulting
     * in at max 1 and at minimum 0 ticks. That's why we use -1 i.e. spacing
     * divided by 10 to get at least 10 and at maximum 11 ticks */
    return spacing;
}

int printPrettyFloat( char * rString, float rValue )
{
    const int nFloatDigits = 3;
    /* unfortunately %g doesn't work optimally. '%.1g' elongates 10 to 1e+01 */
    /* only print up to two 0s, e.g. 0.01, but not 0.001 or
     * 100, but not 1000, instead use 1e-3 and 1e3 respectively */
    int exponent = (int) log10( fabs(rValue) ); /* conversion to int floors! */
    if ( rValue == 0 )
        exponent = 0;

    if ( fabs(exponent) < nFloatDigits )
    {
        int nChars = sprintf( rString, "%.*f", nFloatDigits+3, rValue );
        if (nChars <= 0)
            return nChars;

        /* strip trailing 0s in representation e.g. '1.25000' -> '1.25' */
        for (int i = nChars-1; i > 0; --i )
            if ( rString[i] != '0' )
            {
                if ( rString[i] == '.' )
                    --i;
                rString[i+1] = 0;
                nChars = i+1;
                break;
            }
        return nChars;
    }
    else
        return sprintf( rString, "%.0e", rValue );
}


const int minorTickSize  = 2;
const int normalTickSize = 3;
const int majorTickSize  = 4;
const int arrowHeadSize  = 5;
const int arrowHeadAngle = 25;

int SDL_RenderDrawVerticalAxis
( SDL_Renderer * rpRenderer, const SDL_Rect & rAxes, float y0, float y1 )
{
    const int &x=rAxes.x, &y=rAxes.y, &h=rAxes.h;
    char tickString[64];

    if (y0 >= y1)
        return 1;

    SDL_RenderDrawArrow( rpRenderer, x,y+h, x,y, arrowHeadSize,arrowHeadAngle );
    const float dyTick = chooseTickSpacing(y0,y1);

    /* count how many major and middle tick labels will be written out.
     * If less than 2, then also label minor ticks */
    const int yMiddleTickMin = (int) ceil ( y0/dyTick ) / 5;
    const int yMiddleTickMax = (int) floor( y1/dyTick ) / 5;
    const int nYMiddleTicks  = yMiddleTickMax - yMiddleTickMin + 1;
    int yLabelSkip = 5;
    if ( nYMiddleTicks < 2 )
        yLabelSkip = 1;
    /*
    std::cout << "Draw Vertical Axis ["<<y0<<","<<y1<<"]: tickMin="
              << yMiddleTickMin<<", tickMax="<<yMiddleTickMax<<"\n";
    */

    for ( int iTick = ceil( y0/dyTick ); ; ++iTick )
    {
        const float yTickValue = iTick*dyTick;
        int yTick = y+h /*zero line*/ - (yTickValue-y0)/(y1-y0)*h /*value*/;
        /* stop drawing ticks if arrow head reached on axis */
        if (yTick < y+arrowHeadSize )
            break;

        if ( iTick % 10 == 0 )
            SDL_RenderDrawLine( rpRenderer, x,yTick, x - majorTickSize ,yTick );
        else if ( iTick % 5 == 0 )
            SDL_RenderDrawLine( rpRenderer, x,yTick, x - normalTickSize,yTick );
        else
            SDL_RenderDrawLine( rpRenderer, x,yTick, x - minorTickSize ,yTick );

        if ( iTick % yLabelSkip == 0 or iTick == 0  )
        {
            printPrettyFloat( tickString, yTickValue );
            SDL_Rect target = { x-majorTickSize-2, yTick, 0,0 };
            SDL_drawString( rpRenderer, SDL_PlotFonts::instance()->tickFont,
                tickString, &target, 2/*right x*/,1/*center y*/ );
        }
    }
    return 0;
}

int SDL_RenderDrawHorizontalAxis
( SDL_Renderer * rpRenderer, const SDL_Rect & rAxes, float x0, float x1 )
{
    const int &x=rAxes.x, &y=rAxes.y, &w=rAxes.w;
    char tickString[64];

    if (x0 >= x1)
        return 1;

    SDL_RenderDrawArrow( rpRenderer, x,y, x+w,y, arrowHeadSize,arrowHeadAngle );
    const float dxTick = chooseTickSpacing(x0,x1);

    /* e.g. if y0 = 0.13 and spacing = 0.1 then we would want the first xTick
     * to be at 0.2, meaning iTick would be 2 = ceil(0.13/0.1 = 1.3) */
    /* iTick doesn't start at 0! e.g. if dxTick = 0.1 and x0 = 2.3, then
     * iTick = 23, which is a minor tick, the next major tick will be 30 */
    for ( int iTick = ceil( x0/dxTick ); ; ++iTick )
    {
        const float xTickValue = iTick*dxTick;
        int xTick = x + (xTickValue-x0)/(x1-x0)*w;
        if (xTick >= x+w )
            break;

        if ( iTick % 10 == 0 )
            SDL_RenderDrawLine( rpRenderer, xTick,y, xTick,y+majorTickSize  );
        else if ( iTick % 5 == 0 )
            SDL_RenderDrawLine( rpRenderer, xTick,y, xTick,y+normalTickSize );
        else
            SDL_RenderDrawLine( rpRenderer, xTick,y, xTick,y+minorTickSize  );

        if ( iTick % 5 == 0 or iTick == 0 )
        {
            printPrettyFloat( tickString, xTickValue );
            SDL_Rect target = { xTick, y+majorTickSize+2, 0,0 };
            SDL_drawString( rpRenderer, SDL_PlotFonts::instance()->tickFont,
                tickString, &target, 1/*center x*/,0 );
        }
    }
    return 0;
}

int SDL_RenderDrawAxes
( SDL_Renderer * rpRenderer, const SDL_Rect & rAxes,
  float x0, float x1, float y0, float y1 )
{
    int error = 0;
    error |= SDL_RenderDrawVerticalAxis  ( rpRenderer,rAxes,y0,y1 );
    SDL_Rect xAxis = rAxes;
    xAxis.y += rAxes.h;
    error |= SDL_RenderDrawHorizontalAxis( rpRenderer,xAxis,x0,x1 );
    return error;
}

void SDL_PlotGetYRange
( float * values, int nValues, float * rY0, float * rY1,
  const char * rTitle = "" )
{
    float &y0 = *rY0, &y1 = *rY1;

    /* find minimum and maximum function value for plot range */
    y0 = FLT_MAX;
    y1 = FLT_MIN;
    for ( int i = 0; i < nValues; ++i )
    {
        y0 = fmin( y0, values[i] );
        y1 = fmax( y1, values[i] );
    }

    /* leave a bit of space for the title */
    const float ySpan = y1-y0;
    y0 -= 0.0*ySpan;
    if ( rTitle[0] != 0 )
        y1 += 0.2*ySpan; /* @todo: make this value dependent on labelFontSize */
}

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

int SDL_RenderDrawHistogram
( SDL_Renderer * rpRenderer, const SDL_Rect & rAxes,
  float x0, float x1, float y0, float y1,
  float * values, int nValues,
  int binWidth,  bool fill, bool drawAxis, const char * title )
{
    if ( y0 == y1 )
        SDL_PlotGetYRange( values,nValues, &y0,&y1, title );
    if ( x0 == x1 )
    {
        x0 = 0;
        x1 = nValues-1;
    }
    /* include the 0 centerline in the plot! */
    y0 = fmin( y0, 0.0f );
    y1 = fmax( y1, 0.0f );

    if (x0 >= x1) return 1;
    if (y0 >= y1) return 1;

    /* automatically choose a binWidth which fills the x-Range as best as
     * possible. For many bins this becomes less and less optimal, because
     * binWidth is the number of pixels, meaning an integer. The total width
     * of the plot thefore increases discretely in multiples of nValues pixel!*/
    if ( binWidth <= 0 )
        binWidth = rAxes.w / nValues; /* integer floor division */

    for ( int i = 0; i < nValues; ++i )
    {
        /* convert function value back to pixel by shifting and scaling */
        const int height = ( values[i]-0.0f )/( y1-y0 )*rAxes.h;

        SDL_Rect rect = rAxes;
        rect.x += i*binWidth;
        /* if 0 is on the y-axis, then draw the x-axis at the height of 0,
         * instead of at the bottom */
        if (y0 <= 0 and y1 >= 0)
            rect.y += fabs(y1)/(y1-y0)*rAxes.h /* zero line */;
        /* only need to further move y position, if bin is above 0 */
        if (height >= 0)
            rect.y -= abs(height);
        rect.w  = binWidth;
        rect.h  = abs(height);

        SDL_RenderDrawThickRect( rpRenderer, rect, 1 );
    }

    if ( drawAxis )
    {
        int error = 0;
        error |= SDL_RenderDrawVerticalAxis  ( rpRenderer,rAxes,y0,y1 );
        /* if 0 is on the y-axis, then draw the x-axis at the height of 0,
         * instead of at the bottom */
        if (y0 <= 0 and y1 >= 0)
        {
            SDL_Rect zeroAxis = rAxes;
            zeroAxis.y += fabs(y1)/(y1-y0)*rAxes.h;
            error |= SDL_RenderDrawHorizontalAxis( rpRenderer,zeroAxis,x0,x1 );
        }
        if ( error != 0 )
            return error;
    }

    /* Draw title */
    SDL_Rect titleLoc = { rAxes.x + rAxes.w/2, rAxes.y, 0,0 };
    SDL_drawString( rpRenderer, SDL_PlotFonts::instance()->labelFont,
        title, &titleLoc, 1 /*center*/, 0 /*top align*/ );

    return 0;
}
