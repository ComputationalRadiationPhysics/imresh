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


./convertToCupla.sh ./imresh/tests/imresh/algorithms/testVectorReduce.cpp
./convertToCupla.sh ./imresh/src/imresh/algorithms/vectorElementwise.cpp
./convertToCupla.sh ./imresh/src/imresh/algorithms/vectorElementwise.hpp
./convertToCupla.sh ./imresh/src/imresh/algorithms/vectorElementwise.tpp
./convertToCupla.sh ./imresh/src/imresh/algorithms/vectorReduce.cpp
./convertToCupla.sh ./imresh/src/imresh/algorithms/vectorReduce.hpp
./convertToCupla.sh ./imresh/src/imresh/algorithms/vectorReduce.tpp

ToDo: Remove #if false from testVectorReduce.cpp and include more and more tests ...

compile error cupla matrixMul example with CUDA
    [ 40%] Building NVCC (Device) object CMakeFiles/matrixMul.dir/src/matrixMul_generated_matrixMul.cpp.o
    [ 50%] Building NVCC (Device) object CMakeFiles/matrixMul.dir/__/__/__/src/matrixMul_generated_memory.cpp.o
    /media/d/Studium/9TH SEMESTER/imresh/cupla/alpaka/include/alpaka/stream/Traits.hpp(75): error: incomplete type is not allowed
              detected during:
                instantiation of "auto alpaka::stream::enqueue(TStream &, TTask &&)->void [with TStream=cupla::AccStream, TTask=alpaka::mem::view::cpu::detail::TaskCopy<cupla::HostBufWrapper<1U>, cupla::HostBufWrapper<1U>, alpaka::Vec<cupla::AlpakaDim<1U>, cupla::MemSizeType>>]" 
    /media/d/Studium/9TH SEMESTER/imresh/cupla/alpaka/include/alpaka/mem/view/Traits.hpp(436): here
                instantiation of "auto alpaka::mem::view::copy(TStream &, TViewDst &, const TViewSrc &, const TExtent &)->void [with TExtent=alpaka::Vec<cupla::AlpakaDim<1U>, cupla::MemSizeType>, TViewSrc=cupla::HostBufWrapper<1U>, TViewDst=cupla::HostBufWrapper<1U>, TStream=cupla::AccStream]" 
    /media/d/Studium/9TH SEMESTER/imresh/cupla/src/memory.cpp(331): here

    /media/d/Studium/9TH SEMESTER/imresh/cupla/alpaka/include/alpaka/stream/Traits.hpp(75): error: incomplete type is not allowed
              detected during:
                instantiation of "auto alpaka::stream::enqueue(TStream &, TTask &&)->void [with TStream=cupla::AccStream, TTask=alpaka::mem::view::cpu::detail::TaskCopy<cupla::HostBufWrapper<2U>, cupla::HostBufWrapper<2U>, alpaka::Vec<cupla::AlpakaDim<2U>, cupla::MemSizeType>>]" 
    /media/d/Studium/9TH SEMESTER/imresh/cupla/alpaka/include/alpaka/mem/view/Traits.hpp(436): here
                instantiation of "auto alpaka::mem::view::copy(TStream &, TViewDst &, const TViewSrc &, const TExtent &)->void [with TExtent=alpaka::Vec<cupla::AlpakaDim<2U>, cupla::MemSizeType>, TViewSrc=cupla::HostBufWrapper<2U>, TViewDst=cupla::HostBufWrapper<2U>, TStream=cupla::AccStream]" 
    /media/d/Studium/9TH SEMESTER/imresh/cupla/src/memory.cpp(605): here

    2 errors detected in the compilation of "/tmp/tmpxft_00004d92_00000000-7_memory.cpp1.ii".
    CMake Error at matrixMul_generated_memory.cpp.o.cmake:266 (message):
      Error generating file /media/d/Studium/9TH
      SEMESTER/imresh/cupla/build/CMakeFiles/matrixMul.dir/__/__/__/src/./matrixMul_generated_memory.cpp.o


    CMakeFiles/matrixMul.dir/build.make:5099: recipe for target 'CMakeFiles/matrixMul.dir/__/__/__/src/matrixMul_generated_memory.cpp.o' failed
    make[2]: *** [CMakeFiles/matrixMul.dir/__/__/__/src/matrixMul_generated_memory.cpp.o] Error 1
    CMakeFiles/Makefile2:141: recipe for target 'CMakeFiles/matrixMul.dir/all' failed
    make[1]: *** [CMakeFiles/matrixMul.dir/all] Error 2
    Makefile:83: recipe for target 'all' failed
    make: *** [all] Error 2

Problem getting workdiv ... 

    /media/d/Studium/9TH SEMESTER/imresh/cupla/alpaka/include/alpaka/workdiv/Traits.hpp(71): error: calling a __device__ function("getWorkDiv") from a __host__ __device__ function("getWorkDiv") is not allowed
              detected during:
                instantiation of "auto alpaka::workdiv::getWorkDiv<TOrigin,TUnit,TWorkDiv>(const TWorkDiv &)->alpaka::Vec<alpaka::dim::Dim<TWorkDiv>, alpaka::size::Size<TWorkDiv>> [with TOrigin=alpaka::origin::Block, TUnit=alpaka::unit::Threads, TWorkDiv=alpaka::workdiv::WorkDivCudaBuiltIn<cupla::KernelDim, cupla::IdxType>]" 
    (136): here
                instantiation of "auto alpaka::workdiv::traits::GetWorkDiv<TWorkDiv, alpaka::origin::Block, alpaka::unit::Threads, std::enable_if<<expression>, void>::type>::getWorkDiv(const TWorkDiv &)->alpaka::Vec<alpaka::dim::Dim<TWorkDiv::WorkDivBase>, alpaka::size::Size<TWorkDiv::WorkDivBase>> [with TWorkDiv=alpaka::acc::AccGpuCudaRt<cupla::KernelDim, cupla::IdxType>]" 
    (71): here
                instantiation of "auto alpaka::workdiv::getWorkDiv<TOrigin,TUnit,TWorkDiv>(const TWorkDiv &)->alpaka::Vec<alpaka::dim::Dim<TWorkDiv>, alpaka::size::Size<TWorkDiv>> [with TOrigin=alpaka::origin::Block, TUnit=alpaka::unit::Threads, TWorkDiv=alpaka::acc::AccGpuCudaRt<cupla::KernelDim, cupla::IdxType>]" 
    /media/d/Studium/9TH SEMESTER/imresh/src/imresh/algorithms/vectorElementwise.tpp(59): here
                instantiation of "void imresh::algorithms::cudaKernelApplyHioDomainConstraints<T_COMPLEX, T_PREC>::operator()(const T_ACC &, T_COMPLEX *, const T_COMPLEX *, const T_PREC *, unsigned int, T_PREC) const [with T_COMPLEX=cufftComplex, T_PREC=float, T_ACC=alpaka::acc::AccGpuCudaRt<cupla::KernelDim, cupla::IdxType>]" 
    /media/d/Studium/9TH SEMESTER/imresh/src/imresh/algorithms/vectorElementwise.cpp(54): here

    /media/d/Studium/9TH SEMESTER/imresh/cupla/alpaka/include/alpaka/workdiv/Traits.hpp(71): error: calling a __device__ function("getWorkDiv") from a __host__ __device__ function("getWorkDiv") is not allowed
              detected during:
                instantiation of "auto alpaka::workdiv::getWorkDiv<TOrigin,TUnit,TWorkDiv>(const TWorkDiv &)->alpaka::Vec<alpaka::dim::Dim<TWorkDiv>, alpaka::size::Size<TWorkDiv>> [with TOrigin=alpaka::origin::Grid, TUnit=alpaka::unit::Blocks, TWorkDiv=alpaka::workdiv::WorkDivCudaBuiltIn<cupla::KernelDim, cupla::IdxType>]" 
    (107): here
                instantiation of "auto alpaka::workdiv::traits::GetWorkDiv<TWorkDiv, alpaka::origin::Grid, alpaka::unit::Blocks, std::enable_if<<expression>, void>::type>::getWorkDiv(const TWorkDiv &)->alpaka::Vec<alpaka::dim::Dim<TWorkDiv::WorkDivBase>, alpaka::size::Size<TWorkDiv::WorkDivBase>> [with TWorkDiv=alpaka::acc::AccGpuCudaRt<cupla::KernelDim, cupla::IdxType>]" 
    (71): here
                instantiation of "auto alpaka::workdiv::getWorkDiv<TOrigin,TUnit,TWorkDiv>(const TWorkDiv &)->alpaka::Vec<alpaka::dim::Dim<TWorkDiv>, alpaka::size::Size<TWorkDiv>> [with TOrigin=alpaka::origin::Grid, TUnit=alpaka::unit::Blocks, TWorkDiv=alpaka::acc::AccGpuCudaRt<cupla::KernelDim, cupla::IdxType>]" 
    /media/d/Studium/9TH SEMESTER/imresh/src/imresh/algorithms/vectorElementwise.tpp(61): here
                instantiation of "void imresh::algorithms::cudaKernelApplyHioDomainConstraints<T_COMPLEX, T_PREC>::operator()(const T_ACC &, T_COMPLEX *, const T_COMPLEX *, const T_PREC *, unsigned int, T_PREC) const [with T_COMPLEX=cufftComplex, T_PREC=float, T_ACC=alpaka::acc::AccGpuCudaRt<cupla::KernelDim, cupla::IdxType>]" 
    /media/d/Studium/9TH SEMESTER/imresh/src/imresh/algorithms/vectorElementwise.cpp(54): here


