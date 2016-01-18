# imresh

Shrink-Wrap Imaginar Coefficient Reconstruction Algorithm

## Compilation and Installation

### Prerequisites

To compile this library you need

* C++ Compiler (with `c++11` support)

* CUDA (`7.5+`)

* CMake (`3.3+`)

* OpenMP

* FFTW3 (single precision build)

### Optional Dependencies

* [libSplash](https://github.com/ComputationalRadiationPhysics/libSplash)
    (for reading and writing HDF5)

* [PNGwriter](https://github.com/pngwriter/pngwriter) (for storing
    reconstructed images as PNGs)

### Build options

* `-DCMAKE_INSTALL_PREFIX`

    The prefix to install the library and additional files (such as headers) into.

* `-DRUN_TESTS` (default off)

    If true tests will be run.

* `-DBUILD_EXAMPLES` (default off)

    If true the examples from the examples directory will be built.

* `-DIMRESH_DEBUG` (default off)

    Adds at least debugging symbols to the code.

* `-DBUILD_DOC` (default on)

    Build the Doxygen documentation.

* `-DUSE_PNG` (default off)

    Enable PNG output.

* `-DUSE_SPLASH` (default off)

    Enable HDF5 in- and output.

### Building

1. Create a build directory

        mkdir build
        cd build

2. Invoking CMake

        cmake ..

    To build and run everything, try

        cmake .. -DRUN_TESTS=on -DBUILD_EXAMPLES=on

3. Invoking make

        make

4. Installing

        make install

## Usage

### Basic usage

The usage of _imresh_ is mainly divided into five parts:

1. Library initialization

2. Image loading

3. Image processing

4. Image writing

5. Library deinitialization

where image loading and writing can also be handled outside the library.

1. The library initialization is (from the user's perspective) just a single
    call to `imresh::io::taskQueueInit( )`. Internally this creates
    `cudaStream_t`s for each multiprocessor on each CUDA capable device found
    and stores them for later access.

2. Image loading can be done through _imresh_'s own loading functions (found in
    `imresh::io::readInFuncs`) or with self-written functions.

    > _Note:_

    > Your self-written functions have to provide you both the image dimensions
    > and the host memory containing the image. This memory has to be allocated
    > via `cudaMallocHost` in order to ensure _imresh_'s correct behaviour.

3. Image processing is just a call to `imresh::io::addTask( )` (for explanation
    of the parameters please have a look at the Doxygen). This will start a
    thread (a C++ `std::thread` thread to be precise) handling your data
    transfers and image processing on the least recently used stream available.
    The given data write out function will be called inside of this thread, too.

4. Image writing can, just as the loading, be done via _imresh_'s own write out
    functions (found in `imresh::io::writeOutFuncs`) or with self-written
    functions. These have to match the following signature:

        void writeOutFunc( float* memory, std::pair<unsigned int,unsigned int> size, std::string filename);

    where `memory` is the raw image data, `size` the image dimension
    (`size.first` is horizontal, `size.second` is vertical) and `filename` the
    name of the file to store the image in.

5. Library deinitialization is again just a call to `imresh::io::taskQueueDeinit( )`.
    This will handle stream destroying, memory freeing and so on for you.

    > _Note:_

    > When you're using your own data reading and/or writing functions, you'll
    > have to handle the memory inside of this functions yourself.

## Authors

* Maximilian Knespel (m.knespel at hzdr dot de)

* Philipp Trommler (philipp.trommler at tu-dresden dot de)

## Known Bugs

* `/usr/include/fftw3.h(373): error: identifier "__float128" is undefined`

    Update your fftw library or manually apply the patch shown [here](https://github.com/FFTW/fftw3/commit/07ef78dc1b273a40fb4f7db1797d12d3423b1f40),
    i.e. add `|| defined(__CUDACC__)` to the faulty line in the header.

* `stddef.h(432): error: identifier "nullptr" is undefined`

    Your CMake version is too old.
