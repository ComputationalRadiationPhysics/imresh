#!/bin/bash

#rm -r ./*
cmake ..                                    \
    -DCMAKE_CXX_COMPILER=$(which g++-4.9)   \
    -DCMAKE_C_COMPILER=$(which gcc-4.9)     \
    -DBUILD_DOC=OFF                         \
    -DIMRESH_DEBUG=ON                       \
    -DBUILD_EXAMPLES=ON                     \
    -DRUN_TESTS=ON                          \
    -DALPAKA_ACC_GPU_CUDA_ENABLE=ON         \
    -DBUILD_EXAMPLES=OFF                    \
    -DRUN_TESTS=ON                          \
    -DUSE_PNG=ON                            \
    -DUSE_SPLASH=ON                         \
    -DUSE_TIFF=ON                           \
    -DUSE_FFTW=ON
#make -j 3
