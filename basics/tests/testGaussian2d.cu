
#include "testGaussian2d.h"
#include "cudacommon.h" // this should not be included in the header, because else compile errors will happen, wenn including cudacommon.cu from a .cpp file


namespace imresh {
namespace test {


/**
 * Plots original, horizontally and vertically blurred intermediary steps
 *
 * Also compares the result of the CPU blur with the CUDA blur
 **/
void testGaussianBlur2d
(
  SDL_Renderer * const rpRenderer, SDL_Rect rect, float * const data,
  const unsigned nDataX, const unsigned nDataY, const float sigma,
  const char * const title
)
{
    using namespace imresh::math::image;

    /* do intermediary steps with CUDA */
    const unsigned dataSize = sizeof(float)*nDataX*nDataY;
    float * cudaHBlur = (float*) malloc( dataSize );
    float * cudaBlur  = (float*) malloc( dataSize );
    float * dpData;
    CUDA_ERROR( cudaMalloc( (void**) &dpData, dataSize ) );

    CUDA_ERROR( cudaMemcpy( dpData, data, dataSize, cudaMemcpyHostToDevice ) );
    cudaGaussianBlurHorizontal( dpData, nDataX, nDataY, sigma );
    CUDA_ERROR( cudaMemcpy( cudaHBlur, dpData, dataSize, cudaMemcpyDeviceToHost ) );
    cudaGaussianBlurVertical( dpData, nDataX, nDataY, sigma );
    CUDA_ERROR( cudaMemcpy( cudaBlur, dpData, dataSize, cudaMemcpyDeviceToHost ) );

    /* do intermediary steps using the CPU */
    float * cpuHBlur  = (float*) malloc( dataSize );
    float * cpuBlur   = (float*) malloc( dataSize );
    memcpy( cpuHBlur, data, dataSize );
    gaussianBlurHorizontal( cpuHBlur, nDataX, nDataY, sigma );
    memcpy( cpuBlur, cpuHBlur, dataSize );
    gaussianBlurVertical( cpuBlur, nDataX, nDataY, sigma );

    /* compare results from GPU with CPU */
    using imresh::math::vector::vectorMaxAbsDiff;
    float absMaxErrH = vectorMaxAbsDiff( cudaHBlur, cpuHBlur, nDataX*nDataY );
    std::cout << "Maximum difference after horizontal blurring: " << absMaxErrH << "\n";
    assert( absMaxErrH < 10*FLT_EPSILON );
    float absMaxErr = vectorMaxAbsDiff( cudaBlur, cpuBlur, nDataX*nDataY );
    std::cout << "Maximum difference after blurring: " << absMaxErr << "\n";
    /*assert( absMaxErr < 10*FLT_EPSILON ); */

    float * cpuError  = (float*) malloc( dataSize );
    for ( unsigned i = 0; i < nDataX*nDataY; ++i )
        cpuError[i] = 0.5* fabs( cpuHBlur[i] - cudaHBlur[i] ) / FLT_EPSILON;

    /* plot original image */
    char title2[128];
    SDL_RenderDrawMatrix( rpRenderer, rect, 0,0,0,0, data,nDataX,nDataY,
                          true/*drawAxis*/, title );

    /* plot horizontally blurred image */
    SDL_RenderDrawArrow( rpRenderer, rect.x+1.1*rect.w,rect.y+rect.h/2,
                                     rect.x+1.3*rect.w,rect.y+rect.h/2 );
    rect.x += 1.5*rect.w;
    sprintf( title2,"G_h(s=%0.1f)*%s",sigma,title );
    SDL_RenderDrawMatrix( rpRenderer, rect, 0,0,0,0, cpuBlur,nDataX,nDataY,
                          true/*drawAxis*/, title2 );

    /* plot horizontally blurred image */
    SDL_RenderDrawArrow( rpRenderer, rect.x+1.1*rect.w,rect.y+rect.h/2,
                                     rect.x+1.3*rect.w,rect.y+rect.h/2 );
    rect.x += 1.5*rect.w;
    sprintf( title2,"G_v o G_h(s=%0.1f)*%s",sigma,title );
    SDL_RenderDrawMatrix( rpRenderer, rect, 0,0,0,0, cudaBlur,nDataX,nDataY,
                          true/*drawAxis*/, title2 );

    /* free everything */
    memcpy( data, cpuBlur, dataSize );
    CUDA_ERROR( cudaFree( dpData ) );
    free( cpuBlur   );
    free( cpuHBlur  );
    free( cudaBlur  );
    free( cudaHBlur );
}

void testGaussian2d( SDL_Renderer * const rpRenderer )
{
    /* ideal window size for this test is 1024x640 px */
{
    srand(165158631);
    const unsigned nDataX = 20;
    const unsigned nDataY = 20;
    SDL_Rect rect = { 40,40,5*nDataX,5*nDataY };
    float data[nDataX*nDataY];

    /* Try different data sets */
    /**
     * +--------+        +--------+   # - black
     * |  +    +|        |     .  |   x - gray
     * |     #  |        |p   .i. |   p - lighter gray
     * |#       |   ->   |xo   .  |   o - very light gray
     * |       +|        |p   o   |   i - also light gray
     * |    #   |        |   pxp  |   . - gray/white barely visible
     * +--------+        +--------+     - white
     * Note that the two dots at the borders must result in the exact same
     * blurred value (only rotated by 90°). This is not obvious, because
     * we apply gausian first horizontal, then vertical, but it works and
     * makes the gaussian-N-dimensional algorithm much faster!
     **/
    for ( unsigned i = 0; i < nDataX*nDataY; ++i )
        data[i] = 1.0;
    data[10]           = 0;
    data[10*nDataX]    = 0;
    data[12*nDataX+12] = 0;
    data[nDataY*nDataX-1]  = 0;
    data[nDataY*nDataX-12] = 0;
    data[7*nDataX-1]       = 0;
    testGaussianBlur2d( rpRenderer,rect, data,nDataX,nDataY, 1.0, "3-Points" );
    rect.y += 140;
    assert( data[9] != 1.0 );
    assert( data[9] == data[11] );
    assert( data[9*nDataX] == data[11*nDataX] );
    assert( data[9] == data[11*nDataX] );
    assert( data[nDataX+10] == data[10*nDataX+1] );

    /* same as above in white on black */
    for ( unsigned i = 0; i < nDataX*nDataY; ++i )
        data[i] = 0;
    data[10]           = 1;
    data[10*nDataX]    = 1;
    data[12*nDataX+12] = 1;
    data[nDataY*nDataX-1]  = 1;
    data[nDataY*nDataX-12] = 1;
    data[7*nDataX-1]       = 1;
    testGaussianBlur2d( rpRenderer,rect, data,nDataX,nDataY, 1.0, "3-Points" );
    rect.y += 140;
    assert( data[9] != 0 );
    assert( data[9] == data[11] );
    assert( data[9*nDataX] == data[11*nDataX] );
    assert( data[9] == data[11*nDataX] );
    assert( data[nDataX+10] == data[10*nDataX+1] );

    /* blur a random image (draw result to the right of above images) */
    rect.x += (3*1.5+1)*(5*nDataX);
    rect.y  = 20;
    for ( unsigned i = 0; i < nDataX*nDataY; ++i )
        data[i] = rand()/(double)RAND_MAX;
    testGaussianBlur2d( rpRenderer,rect, data,nDataX,nDataY, 1.0, "Random" );
    rect.y += 140;
    for ( unsigned i = 0; i < nDataX*nDataY; ++i )
        data[i] = rand()/(double)RAND_MAX;
    testGaussianBlur2d( rpRenderer,rect, data,nDataX,nDataY, 2.0, "Random" );
    rect.y += 140;
}
{
    /* try with quite a large image! */
    srand(165158631);
    const unsigned nDataX = 240;
    const unsigned nDataY = 240;
    SDL_Rect rect = { 30,320,nDataX,nDataY };
    float data[nDataX*nDataY];

    /* fill with random data */
    for ( unsigned i = 0; i < nDataX*nDataY; ++i )
        data[i] = rand()/(double)RAND_MAX;
    testGaussianBlur2d( rpRenderer,rect, data,nDataX,nDataY, 3.0, "Random" );
}
}


} // namespace imresh
} // namespace test
