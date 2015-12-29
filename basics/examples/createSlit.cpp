
namespace imresh {
namespace examples {


float * createVerticalSingleSlit( const unsigned Nx, const unsigned Ny )
{
    using namespace imresh::math::image;

    float * data = new float[Nx*Ny];
    memset( data, 0, Nx*Ny*sizeof(float) );

    const int slitHalfHeight = (int) ceilf( 0.3*Nx );
    const int slitHalfWidth  = (int) ceilf( 0.1*Nx );
    for ( unsigned iy = Ny/2 - slitHalfHeight+1; iy < Ny/2 + slitHalfHeight; ++iy )
    for ( unsigned ix = Nx/2 - slitHalfWidth +1; ix < Nx/2 + slitHalfWidth ; ++ix )
        data[iy*Nx + ix] = 1;

    return data;
}


} // namespace examples
} // namespace imresh
