#!/bin/bash

#rm -r ./*
cmake ..                                        \
    -DCMAKE_CXX_COMPILER=$(which g++-4.9)       \
    -DCMAKE_C_COMPILER=$(which gcc-4.9)         \
    -DBUILD_DOC=OFF                             \
    -DIMRESH_DEBUG=ON                           \
    -DBUILD_EXAMPLES=ON                         \
    -DRUN_TESTS=ON                              \
    -DUSE_PNG=ON                                \
    -DUSE_SPLASH=ON                             \
    -DUSE_FFTW=OFF                              \
    -DUSE_TIFF=ON
