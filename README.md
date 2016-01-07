# imresh

Shrink-Wrap Imaginar Coefficient Reconstruction Algorithm

## Compilation and Installation

### Prerequisites

To compile this library you need

* C++ Compiler (with `c++11` support)

* CUDA (`7.5+`)

* CMake (`3.3+`)

* [libSplash](https://github.com/ComputationalRadiationPhysics/libSplash)

* OpenMP

* FFTW3 (single precision build)

### Build options

* `-DCMAKE_INSTALL_PREFIX`

    The prefix to install the library and additional files (such as headers) into.

* `-DRUN_TESTS` (default off)

    If true tests will be run.

* `-DBUILD_EXAMPLES` (default off)

    If true the examples from the examples directory will be built.

* `-DIMRESH_DEBUG` (default off)

    Adds at least debugging symbols to the code.

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

> TODO

## Authors

* Maximilian Knespel (m.knespel at hzdr dot de)

* Philipp Trommler (philipp.trommler at tu-dresden dot de)

## Known Bugs

* `/usr/include/fftw3.h(373): error: identifier "__float128" is undefined`

    Update your fftw library or manually apply the patch shown [here](https://github.com/FFTW/fftw3/commit/07ef78dc1b273a40fb4f7db1797d12d3423b1f40),
    i.e. add `|| defined(__CUDACC__)` to the faulty line in the header.

* `stddef.h(432): error: identifier "nullptr" is undefined`

    Your CMake version is too old.
