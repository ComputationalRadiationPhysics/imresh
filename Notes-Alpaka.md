### cupla example:

    cd imresh/build/
    export IMRESH_ROOT=
    cmake .. -DIMRESH_DEBUG=ON -DCMAKE_CXX_COMPILER=$(which g++-4.9) -DCMAKE_C_COMPILER=$(which gcc-4.9) -DBUILD_DOC=OFF -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE=ON

### How to explicitely instantiate templates:

from cupla/types.h

    #ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
        using Acc = ::alpaka::acc::AccCpuOmp2Threads<
            KernelDim,
            IdxType
        >;
    #endif

    #ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
        using Acc = ::alpaka::acc::AccCpuOmp2Blocks<
            KernelDim,
            IdxType
        >;
    #endif

    #ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
        using Acc = ::alpaka::acc::AccCpuThreads<
            KernelDim,
            IdxType
        >;
    #endif

    #ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
        using Acc = ::alpaka::acc::AccCpuSerial<
            KernelDim,
            IdxType
        >;
    #endif

  => that Acc can be used for `T_ACC`

Convert vectorElementwise.hpp:

    sed 's/__global__ void /struct /'

convert .cpp:

    sed 's/__global__/template< class T_ACC >\nALPAKA_FN_ACC/'

matrixmul.cpp

    error = cuda Malloc((void **) &d_B, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

  => ist das nicht doppelt gemoppelt? wird doch schon in alpaka gemacht?

### What happens when defining more than one accelerator in alpaka?

### Add support for libraries in alpaka

alpakaAddLibrary


    ./imresh/src/imresh/algorithms/alpaka/vectorElementwise.cpp:94:5: error: explicit instantiation shall not use 'inline' specifier [-fpermissive]
         ALPAKA_FN_ACC void                                      \
         ^
    ./imresh/src/imresh/algorithms/alpaka/vectorElementwise.cpp:104:5: note: in expansion of macro 'INSTANTIATE_TMP'
         INSTANTIATE_TMP( cufftComplex, float )
         ^

  => in alpaka/core/Common.hpp delete all `__force_inline__` and `inline` in lines 39 to 54

> add the qualifier const to each parameter which is not changed inside the kernel
 -> Is this optional?

@TODO:
  - comment atomicCAS in again
  - test __float_as_int implementation
