
void SDL_check(int errorCode, const char * file, int line )
{
    if ( errorCode != 0 )
    {
        std::cout
        << "SDL error in " << file << " line:" << line << " : "
        << SDL_GetError() << "\n" << std::flush;
        assert(false);
    }
}
void SDL_check(const void * ptr, const char * file, int line )
{ if ( ptr == NULL ) SDL_check(1,file,line); }



void checkTTFPtr(void const * ptr, const char * file, int line )
{
    if ( ptr == NULL )
    {
        std::cout
        << "TTF error in " << file << " line:" << line << " : "
        << TTF_GetError() << "\n";
        assert(false);
    }
}


int SDL_SaveRenderedToBmp
( SDL_Renderer * rRenderer, SDL_Window* rpWindow, const char * rFilename )
{
    int w,h;
    SDL_GetRendererOutputSize(rRenderer,&w,&h);

    SDL_Surface *saveSurface = SDL_CreateRGBSurface( 0, w, h, 32,
        /* channel bitmasks */ 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000 );
    SDL_RenderReadPixels( rRenderer, NULL, SDL_PIXELFORMAT_ARGB8888,
        saveSurface->pixels, saveSurface->pitch );
    int errorCode = SDL_SaveBMP( saveSurface,rFilename );
    SDL_FreeSurface(saveSurface);

    return errorCode;
}

void calcIntersectPoint2Lines
( Line2Df const & l0, Line2Df const & l1, Point2Df * intsct )
{
    /* Test if they are even intersecting */
    /*Point2Df v,w,r; float s,u;
    v = l0.p[1] - l0.p[0]; v /= v.norm();
    w = l1.p[1] - l1.p[0]; w /= w.norm();
    r = l1.p[0] - l0.p[0];
    u = v.x*w.y - v.y*w.x; // |v cross w|
    s =  ( w.x*r.y - w.y*r.x )/u; // |w cross r|/u
    //t = -( r.x*v.y - r.y*v.x )/u;
    */
    const Point2Df & p0 = l0.p[0];
    const Point2Df & p1 = l1.p[0];
    Point2Df v0 = l0.p[1] - p0;
    Point2Df v1 = l1.p[1] - p1;
    const float s =   ( v1.y*(p1.x-p0.x) + (p0.y-p1.y)*v1.x ) /
                            ( v0.x*v1.y  -  v0.y*v1.x );
    const float t = - ( v0.y*(p1.x-p0.x) + (p0.y-p1.y)*v0.x ) /
                            ( v1.x*v0.y  -  v1.y*v0.x );
    /* note: comparison with NaN always false */
    if ( s >= 0 and s <= 1 and t >= 0 and t <= 1 )
        *intsct = p0 + v0*s;
        //*intsct = p1 + v1*t;
}

void calcIntersectPoint2Lines
( Line2D const & l0, Line2D const & l1, Point2Df * intsct )
{
    const Line2Df l0f = Line2Df{ (float) l0.p[0].x, (float) l0.p[0].y,
                                 (float) l0.p[1].x, (float) l0.p[1].y };
    const Line2Df l1f = Line2Df{ (float) l1.p[0].x, (float) l1.p[0].y,
                                 (float) l1.p[1].x, (float) l1.p[1].y };
    calcIntersectPoint2Lines( l0f,l1f,intsct );
}

void SDL_RenderDrawCircle
( SDL_Renderer * renderer, const float & cx, const float & cy,
  const float & r, bool fill )
{
    for ( int h=ceil(r); h > -ceil(r); --h )
    {
        const int s = floor(sqrt(r*r-h*h));
        SDL_RenderDrawLine( renderer, cx-s, cy+h, cx+s, cy+h );
    }
}

#if false
float circleIntegral(float x)
{ return 0.5f*(sqrt(1.0f-x*x)*x+asin(x)); }

/* draws filled circle. cx and cy are in the center of the specified pixel,
 * meaning r<=0.5 will result in only 1px set */
void SDL_RenderDrawCircle
( SDL_Renderer * rpRenderer, const float & cx, const float & cy,
  const float & R, bool fill = true )
{
    if (R <= 0.5) return;

    SDL_BlendMode oldBlendMode;
    SDL_GetRenderDrawBlendMode( rpRenderer,&oldBlendMode );
    SDL_SetRenderDrawBlendMode( rpRenderer,SDL_BLENDMODE_BLEND );
    Uint8 cr,cg,cb,ca;
    SDL_GetRenderDrawColor( rpRenderer, &cr,&cg,&cb,&ca );

    /* Note: for r<=0.5 ymax will be 0, meaning the for loop goes from 0 to 0
     * meaning only 1 pixel at maximum will be drawn */
    const int ymax = ceil(R-0.5);
    for ( int y = ymax; y >= -ymax; --y )
    {
        using std::abs;   using std::sqrt;
        using std::floor; using std::ceil;
        using std::min;   using std::max;

        const int s = floor(sqrt(R*R-y*y));
        SDL_SetRenderDrawColor( rpRenderer, cr,cg,cb,255 );
        SDL_RenderDrawLine( rpRenderer, cx-s,cy+y, /* -> */ cx+s,cy+y );

        /* 1 special case is y=0, because in that case the integral is 0,
         * because we don't change x-positions, only y-positions in the limits
         * ! */
        assert( R > 0.5 ); // else abs(y)-0.5 could be problematic too
        float x0 = R*R - (abs(y)+0.5)*(abs(y)+0.5);
        float x1 = R*R - (abs(y)-0.5)*(abs(y)-0.5);
        if ( x0 < 0 ) x0 = 0;
        if ( x1 < 0 ) x1 = 0;
        x0 = sqrt(x0);  // left  global integration limit
        x1 = sqrt(x1);  // right global integration limit
        assert( x1 > x0 );
        const int xmin = (int)floor(x0+0.5); // left  loop limit
        const int xmax = (int)ceil (x0-0.5); // right loop limit

        std::cout << "[y="<<y<<"] x="<<x0<<"~"<<((int)floor(x0)) << ".." << ((int)ceil(x1))<<"~"<<x1<<": \n" << std::flush;

        if (y != 0) for ( int x = xmin; x <= xmax; ++x )
        {
            const float xLeft  = max( x0, (float)x   );
            const float xRight = min( x1, (float)x+1 );
            std::cout << "  ["<<xLeft<<","<<xRight<<"] " << std::flush;
            float area = R*circleIntegral(xRight/R) - R*circleIntegral(xLeft /R);
            #ifndef NDEBUG
                const float areaMin = (xRight-xLeft)*(y-0.5);
                const float areaMax = (xRight-xLeft)*(y+0.5);
                assert( xRight-xLeft >= 0 and xRight-xLeft <= 1 );
                assert( sqrt(R*R-xRight*xRight) - sqrt(R*R-xLeft*xLeft) >= 0 );
                assert( sqrt(R*R-xRight*xRight) - sqrt(R*R-xLeft*xLeft) <= 1 );
                assert( areaMax >= area and areaMin <= areaMin );
            #endif
            //                 - (xRight-xLeft)*(y-0.5);
            const int alpha  = (int)(255*area);
            std::cout << area << "\n" << std::flush;
            assert( alpha <= 255 );
            SDL_SetRenderDrawColor( rpRenderer, cr,cg,cb,alpha );
            SDL_RenderDrawPoint( rpRenderer, cx-x,cy+y );
        }
        std::cout << "\n";
    }

    SDL_SetRenderDrawColor( rpRenderer, cr,cg,cb,ca );
    SDL_SetRenderDrawBlendMode( rpRenderer,oldBlendMode );
}

#endif

void SDL_RenderDrawThickRect
( SDL_Renderer * rpRenderer, const SDL_Rect & rRect, const int rWidth )
{
    assert( rRect.w - 2*rWidth >= 0 );
    int x=rRect.x, y=rRect.y, w=rRect.w, h=rRect.h;

    for ( int i = 0; i < rWidth; ++i )
    {
        SDL_RenderDrawRect ( rpRenderer, &rRect );

        /* @see https://bugzilla.libsdl.org/show_bug.cgi?id=3182 */
        SDL_Point points[4];
        points[0].x = x;
        points[0].y = y;
        points[1].x = x+w-1;
        points[1].y = y;
        points[2].x = x;
        points[2].y = y+h-1;
        points[3].x = x+w-1;
        points[3].y = y+h-1;
        SDL_RenderDrawPoints( rpRenderer, points, 4 );

        x += 1;
        y += 1;
        w -= 2;
        h -= 2;
    }
}

int SDL_RenderDrawArrow
( SDL_Renderer* rpRenderer, int x1, int y1, int x2, int y2,
  float rHeadSize, float rAngle )
{
    SDL_RenderDrawLine( rpRenderer, x1,y1, x2,y2 );
    float ds = tan(30.*M_PI/180.) * rHeadSize;

    #if false
    /* for phi=0, meaning the arrow points to the right : dy=ds, dx=rHeadSize
     * if that is not the case, we need to rotate these two */
    float phi = atan( (float)(y2-y1) / (float)(x2-x1) );
    float dxL = -rHeadSize;
    float dxR = -rHeadSize;
    float dyL = +ds; /* y shift for the upper arrow stroke (left  of arrow) */
    float dyR = -ds; /* y shift for the upper arrow stroke (right of arrow) */

    float dxLT =
    #endif

    /* (x1,y1).(x2,y2) != 0 => y2 = x1/y1*x2 => found orthogonal to our line */
    float x,y;
    x = x2-x1;
    y = y2-y1;
    float dsline = sqrt( x*x + y*y );
    x = x/dsline*rHeadSize;
    y = y/dsline*rHeadSize;

    /* vector orthogonal to drawn line */
    float xp,yp;
    if ( y == 0 )
        xp = 0, yp = x;
    else
        xp = 1, yp = -x/y;
    float dsperp = sqrt( xp*xp + yp*yp );
    xp = xp/dsperp*ds;
    yp = yp/dsperp*ds;

    SDL_RenderDrawLine( rpRenderer, x2,y2, x2-x+xp, y2-y+yp );
    SDL_RenderDrawLine( rpRenderer, x2,y2, x2-x-xp, y2-y-yp );

    return 0;
}


float classSDL_RenderDrawThickLine::f(float x, float a, float b)
{
    assert( a >= 0 and b >= 0 );
    if      ( x < -(a/2+b) ) return 0;
    else if ( x < - a/2    ) return ( x+(a/2+b) )/b;
    else if ( x < + a/2    ) return 1;
    else if ( x < +(a/2+b) ) return ( x-(a/2+b) )/b;
    else                     return 0;
}

float classSDL_RenderDrawThickLine::F(float x, float a, float b)
{
    assert( a >= 0 and b >= 0 );
    if      ( x < -(a/2+b) ) return 0;
    else if ( x < - a/2    ) return  pow( x+(a/2+b) ,2 )/( 2*b );
    else if ( x < + a/2    ) return (a+b)/2+x;
    else if ( x < +(a/2+b) ) return -pow(-x+(a/2+b) ,2 )/( 2*b ) + a+b;
    else                     return a+b;
}

void classSDL_RenderDrawThickLine::operator()
( SDL_Renderer * rpRenderer, int x0, int y0, int x1, int y1, float width )
{
    /* point 0 is supposed to be to the left for some optimizations */
    if ( x1 < x0 )
    {
        std::swap(x0,x1);
        std::swap(y0,y1);
    }

    SDL_BlendMode oldBlendMode;
    SDL_GetRenderDrawBlendMode( rpRenderer,&oldBlendMode );
    SDL_SetRenderDrawBlendMode( rpRenderer,SDL_BLENDMODE_BLEND );
    Uint8 cr,cg,cb,ca;
    SDL_GetRenderDrawColor( rpRenderer, &cr,&cg,&cb,&ca );

    /* set render clip which ensures that we only traverse points we actually
     * may need to set */
    int xmin,ymin,xmax,ymax;
    SDL_GetRendererOutputSize( rpRenderer, &xmax,&ymax );

    xmin = ceil ( std::max( 0.0f     , std::min(x0,x1)-width ) );
    xmax = floor( std::min( xmax-1.0f, std::max(x0,x1)+width ) );
    /* y not yet working with gradient ... */
    ymin = std::max( 0     , std::min(y0,y1) );
    ymax = std::min( ymax-1, std::max(y0,y1) );

    const float phi = atan( (float)(y1-y0)/(x1-x0) );
    const float t   = abs(width/cos(phi));
    const float b   = abs(tan(phi));
    const float a   = t-2*b;

    std::cout << "Drawing thick line: ";

    struct fstruct {
        float a,b;
        fstruct(float ra, float rb) : a(ra), b(rb) {};
        float operator()(float x)
        {
            assert( a >= 0 and b >= 0 );
            if      ( x < -(a/2+b) ) return 0;
            else if ( x < - a/2    ) return  pow( x+(a/2+b) ,2 )/( 2*b );
            else if ( x < + a/2    ) return (a+b)/2+x;
            else if ( x < +(a/2+b) ) return -pow(-x+(a/2+b) ,2 )/( 2*b ) + a+b;
            else                     return a+b;
        }
    } flambda(a,b);
//    SDL_RenderDrawFunction( rpRenderer, flambda, -a/2-b-1, a/2+b+1, 0, a+b,
//      SDL_Rect{ 50,50,200,200 } );

    for ( int y = ymin; y <= ymax; ++y )
    {
        bool didDraw = false;
        const float xline = x0 + (float)(y-std::min(y0,y1))/(y1-y0) * (x1-x0);
        for ( int x = xmin; x <= xmax; ++x )
        {
            float area = F(x-xline+0.5,a,b) - F(x-xline-0.5,a,b);
            printf("[x=%i,y=%i => x'=%f] area=%f\n",x,y,x-xline+0.5,area);
            if ( area == 0 and didDraw ) break;
            else if ( area == 0 ) ++xmin;
            else
            {
                //assert( int(area*255) <= 255 and area >= 0 );
                SDL_SetRenderDrawColor( rpRenderer, cr,cg,cb, area*255 );
                SDL_RenderDrawPoint( rpRenderer, x,y );
                didDraw = true;
            }
        }
    }

    SDL_SetRenderDrawColor( rpRenderer, cr,cg,cb,ca );
    SDL_SetRenderDrawBlendMode( rpRenderer,oldBlendMode );
}

void SDL_RenderDrawThickLine2
( SDL_Renderer * rpRenderer, int x0, int y0, int x1, int y1, float width )
{
    /* Temporarily save blendmode, set to transpaency and save calling colors */
    SDL_BlendMode oldBlendMode;
    SDL_GetRenderDrawBlendMode( rpRenderer,&oldBlendMode );
    SDL_SetRenderDrawBlendMode( rpRenderer,SDL_BLENDMODE_BLEND );
    Uint8 cr,cg,cb,ca;
    SDL_GetRenderDrawColor( rpRenderer, &cr,&cg,&cb,&ca );

    /* because we don't know if x0 < x1 and y0 < y1 in general we need to
     * somehow formulate this abstractly using dx, dy */
    int dx = sgn(x1-x0);
    int dy = sgn(y1-y0);
    // note if dx or dy==0, then x0==x1 meaning the do while loop will be exited
    // after only 1 run

    std::cout << "dx=" << dx << ",dy=" << dy << "\n";

    using std::abs;
    // sample all pixels no matter the orientation of the line
    for ( int y = y0; (dy > 0 ? y <= y1 : y >= y1); y += dy )
    {
        // find x to start at by calculating the intersection:
        for ( int x = x0; (dx > 0 ? x <= x1 : x >= x1); x += dx )
        {
            /* calculate pixel overlap with line */
            double area = 1;

            /* Draw Pixel */
            SDL_SetRenderDrawColor( rpRenderer, cr,cg,cb, area*255 );
            SDL_RenderDrawPoint( rpRenderer, x,y );

            /* break needed for pure horizontal and vertical lines, or else
             * endless loop, because of x+=dx */
            if ( dx == 0 ) break;
        }
        if ( dy == 0 ) break;
    }

    /* Revert changes made to blend mode and color */
    SDL_SetRenderDrawColor( rpRenderer, cr,cg,cb,ca );
    SDL_SetRenderDrawBlendMode( rpRenderer,oldBlendMode );
}

#include <ctime>
const char * getTimeString( void )
{
    time_t date;
    const int nBufferChars = 128;
    static char buffer[nBufferChars];
    struct tm* tm_info;

    time(&date);
    tm_info = localtime(&date);

    strftime(buffer, nBufferChars, "%Y-%m-%d_%H-%M-%S", tm_info);
    return buffer;
}

int SDL_basicControl(SDL_Event const & event, SDL_Window * rpWindow, SDL_Renderer * rpRenderer )
{
    switch ( event.type )
    {
        case SDL_QUIT:
            return 1;
            break;
        case SDL_KEYDOWN:
            switch ( event.key.keysym.sym )
            {
                case SDLK_F5:
                {
                    char filename[256];
                    sprintf( filename, "%s.bmp", getTimeString() );
                    int err = SDL_SaveRenderedToBmp( rpRenderer,rpWindow,filename );
                    if (err == 0)
                        std::cout << "Saved screenshot to " << filename << "\n";
                    else
                        std::cout << "Couldn't save "<<filename<<" does the target folder exist and has the correct permissions?\n";
                    break;
                }
                case SDLK_q:
                case SDLK_ESCAPE:
                    return 1;
                default: break;
            }
            break;
        default: break;
    }
    return 0;
}

template<class T_ANIMATION>
int SDL_animControl( SDL_Event const & event, T_ANIMATION & anim )
{
    switch ( event.type )
    {
        case SDL_KEYDOWN:
            switch ( event.key.keysym.sym )
            {
                case SDLK_s:
                    anim.step();
                    break;
                case SDLK_PLUS:
                    anim.getRenderFrameDelay() /= 2;
                    break;
                case SDLK_MINUS:
                    anim.getRenderFrameDelay() *= 2;
                    if ( anim.getRenderFrameDelay() <= 0 )
                        anim.getRenderFrameDelay() = 1;
                    break;
                case SDLK_SPACE:
                    anim.togglePause();
                    break;
                default: break;
            }
            break;
        default: break;
    }
    return 0;
}

int SDL_drawLineControl::operator()( SDL_Event const & event, SDL_Renderer * rpRenderer )
{
    if ( event.type == SDL_MOUSEBUTTONDOWN )
    {
        SDL_Keymod mod = SDL_GetModState();
        bool shiftPressed = mod & KMOD_LSHIFT;
        if ( p0.x < 0 or not shiftPressed )
        {
            p0 = Point2D{ event.button.x, event.button.y };
            std::cout << "1st point set to (" << p0.x << "," << p0.y << ")\n";
        }
        else if ( shiftPressed )
        {
            Point2D p1 = p0;
            p0 = Point2D{ event.button.x, event.button.y };
            std::cout << "new point set to (" << p0.x << "," << p0.y << ")\n";

            SDL_RenderDrawArrow( rpRenderer, p1.x,p1.y, p0.x,p0.y, 10 );
            Point2Df intsct;
            const Line2D l1 = Line2D{p0,p1};
            if ( l0.p[0].x >= 0 )
            {
                calcIntersectPoint2Lines(l0, l1, &intsct );
                std::cout << "Intersection between line " << l0 << " and " << l1 << " is at (" << intsct.x << "," << intsct.y << ")\n";
                SDL_SetRenderDrawColor( rpRenderer, 0,200,127,128 );
                SDL_RenderDrawCircle( rpRenderer, intsct.x, intsct.y, 3.5 );
                SDL_SetRenderDrawColor( rpRenderer, 255,0,0,255 );
            }
            l0 = l1;
            //p1 = Point2D{-1,-1};
        }
        else if ( not shiftPressed )
            p0 = Point2D{ event.button.x, event.button.y };
        return 1;
    }
    return 0;
}
