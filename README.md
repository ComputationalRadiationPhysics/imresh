# imresh

Shrink-Wrap Imaginar Coefficient Reconstruction Algorithm

## Compilation and Installation

### Prerequisites

To compile this library you need

* A C++ Compiler

* CMake

### Build options

* `-DCMAKE_INSTALL_PREFIX`

    The prefix to install the library and additional files (such as headers) into.

### Building

1. Create a build directory

        mkdir build
        cd build

2. Invoking CMake

        cmake ..

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

* "/usr/include/fftw3.h(373): error: identifier "__float128" is undefined"

  Update your fftw library or manually apply the patch shown here: https://github.com/FFTW/fftw3/commit/07ef78dc1b273a40fb4f7db1797d12d3423b1f40
  i.e. add "|| defined(__CUDACC__)" to the faulty line in the header.
