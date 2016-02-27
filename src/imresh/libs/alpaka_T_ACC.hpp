/* alpaka boiler plate code to choose T_ACC */

#ifndef T_ACC
#   ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
#      define T_ACC ::alpaka::acc::AccCpuOmp2Threads< ::alpaka::dim::DimInt<3u>, unsigned int >
#   endif
#   ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
#      define T_ACC ::alpaka::acc::AccCpuOmp2Blocks< ::alpaka::dim::DimInt<3u>, unsigned int >
#   endif
#   ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
#      define T_ACC ::alpaka::acc::AccCpuThreads< ::alpaka::dim::DimInt<3u>, unsigned int >
#   endif
#   ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#      define T_ACC ::alpaka::acc::AccCpuSerial< ::alpaka::dim::DimInt<3u>, unsigned int >
#   endif
#   ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#      define T_ACC = ::alpaka::acc::AccGpuCudaRt< ::alpaka::dim::DimInt<3u>, unsigned int >
#   endif
#endif
