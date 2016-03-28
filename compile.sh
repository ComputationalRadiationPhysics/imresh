#!/bin/bash

#rm -r ./*
cmake ..                                        \
    -DCMAKE_CXX_COMPILER=$(which g++-4.9)       \
    -DCMAKE_C_COMPILER=$(which gcc-4.9)         \
    -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE=ON     \
    -DBUILD_DOC=OFF                             \
    -DIMRESH_DEBUG=ON                           \
    -DBUILD_EXAMPLES=ON                         \
    -DRUN_TESTS=ON                              \
    -DUSE_PNG=ON                                \
    -DUSE_SPLASH=ON                             \
    -DUSE_TIFF=ON                               \
    -DUSE_FFTW=ON
#make -j 3
#    -DALPAKA_ACC_GPU_CUDA_ENABLE=ON            \
