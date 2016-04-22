cmake_minimum_required(VERSION 3.3.0)
project(imresh)

# Options
option(RUN_TESTS      "Runs some unit tests including some benchmarks" OFF)
option(BUILD_EXAMPLES "Builds some examples showing how to make use of shrinkWrap(...)" OFF)
option(IMRESH_DEBUG   "Enables debugging code, especially many asserts to check for correctness" OFF)
option(BUILD_DOC      "Builds Doxygen Documentation" ON)
option(USE_PNG        "Enables PNG output of reconstructed image" OFF)
option(USE_SPLASH     "Enables HDF5 input and output of images" OFF)
option(USE_FFTW       "If true a CPU reference version is compiled also. It is used internally for benchmarks and error checking tests, but it can also be used externally" OFF)

# General definitions
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Finding packages, calling FindX.cmake and so on
find_package(OpenMP)
find_package(Threads REQUIRED)

# Find cupla
SET( cupla_ROOT "$ENV{CUPLA_ROOT}" CACHE STRING "The location of the cupla library" FORCE )
if( NOT EXISTS "${cupla_ROOT}" )
    set( cupla_ROOT "${PROJECT_SOURCE_DIR}/cupla" )
    if( NOT EXISTS "${cupla_ROOT}/include/cuda_to_cupla.hpp" )
        execute_process(
            COMMAND git submodule update --init -- ./cupla
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
    endif()
    if( NOT EXISTS "${cupla_ROOT}/include/cuda_to_cupla.hpp" )
        message( FATAL_ERROR "Couldn't find '${cupla_ROOT}/include/cuda_to_cupla.hpp" )
    endif()
    set( ENV{CUPLA_ROOT} "${cupla_ROOT}")
endif()
list( APPEND CMAKE_MODULE_PATH "${cupla_ROOT}" )  # add findCupla.cmake
find_package( "cupla" REQUIRED )
add_definitions( ${cupla_DEFINITIONS} )

# Depending on chose alpaka-configuration decide if FFTW needs to be loaded
if(USE_FFTW OR NOT ALPAKA_ACC_GPU_CUDA_ENABLE)
    find_package(FFTW REQUIRED)
    message( "find_package(FFTW) = ${FFTW_LIBRARIES}" )
endif()

find_package( Boost 1.56.0 COMPONENTS system REQUIRED ) # 1.56 needed for boost_unused_parameter
list( APPEND HaLT_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} )
list( APPEND HaLT_INCLUDE_DIRS "${_HALT_ROOT_DIR}/include" )
list( APPEND HaLT_LIBRARIES    ${Boost_LIBRARIES} )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
######

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
        message(FATAL_ERROR "GCC version must be at least 4.7 for -std=c++11 option!")
    endif()
endif()

if(IMRESH_DEBUG)
    # don't set -std=c++11 here! @see
    # https://github.com/ComputationalRadiationPhysics/alpaka/issues/186
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -Wno-unused-parameter -O2 -g -fPIC -pthread ${OpenMP_CXX_FLAGS}")
    add_definitions("-DIMRESH_DEBUG")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -Wall,-Wextra,-Wshadow -G -lineinfo") #  -Xptxas=-v
else()
    # don't set -std=c++11 here! @see
    # https://github.com/ComputationalRadiationPhysics/alpaka/issues/186
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG -fPIC -pthread ${OpenMP_CXX_FLAGS}")
endif()

# add files and compile
set( SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/imresh/algorithms/vectorReduce.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/algorithms/cuda/cudaVectorReduce.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/libs/cudacommon.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/libs/CudaKernelConfig.cpp
)
include_directories(
    ${PROJECT_SOURCE_DIR}/src/imresh
)
include_directories(
    SYSTEM
    ${cupla_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenMP_INCLUDE_DIRS}
)

# Always add all files to the target executable build call to add them to the build project.
alpaka_add_library(
    ${PROJECT_NAME}
    ${SOURCE_FILES}
    ${cupla_SOURCE_FILES}
)
set( LIBRARIES_NEEDED ${LIBRARIES_NEEDED}
    # abstraction layers
    ${_cupla_LINK_LIBRARIES_PUBLIC}
    # runtime libraries (note that normal only either FFTW_LIBRARIES or CUDA_CUFFT_LIBRARIES needs to be set
    ${CUDA_LIBRARIES}
    # OS relevant libraries
    ${OpenMP_LIBRARIES}
)
target_link_libraries( ${PROJECT_NAME} ${LIBRARIES_NEEDED}  )
set( IMRESH_LIBS ${PROJECT_NAME} ${LIBRARIES_NEEDED} )
#install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

# Tests and Benchmarks
include_directories( ${PROJECT_SOURCE_DIR}/tests )
if(RUN_TESTS)
    file( GLOB_RECURSE BENCHMARK_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/benchmark/imresh/algorithms/cuda/cudaVectorReduce.cpp
        ${PROJECT_SOURCE_DIR}/benchmark/imresh/algorithms/cuda/cudaVectorReduce.hpp
        ${PROJECT_SOURCE_DIR}/benchmark/imresh/algorithms/cuda/cudaVectorReduce.tpp
    )
    alpaka_add_library( "benchmark" ${BENCHMARK_SOURCE_FILES} STATIC )
    target_link_libraries("benchmark" ${})

    # reset cxx flags to delete the -DNDEBUG flag, because tests always need to assert
    set(CMAKE_CXX_FLAGS "-O2 -Wall -Wextra -Wunused-parameter -g ${OpenMP_CXX_FLAGS}")
    # for includes of #include "benchmarks/...". Note that
    #    clude_directories(${PROJECT_SOURCE_DIR})
    # would lead to ambiguous includes
    include_directories(${PROJECT_SOURCE_DIR})
    #alpaka_include_directories(${PROJECT_SOURCE_DIR})

    file( GLOB_RECURSE TEST_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/tests/benchmarkHelper.cpp
        ${PROJECT_SOURCE_DIR}/tests/mainTestAll.cpp
    )
    alpaka_add_library( "tests" ${TEST_SOURCE_FILES} STATIC )
    target_link_libraries("tests" ${IMRESH_LIBS} "benchmark")

    alpaka_add_executable("mainTestAll" ${PROJECT_SOURCE_DIR}/tests/mainTestAll.cpp )
    target_link_libraries("mainTestAll" ${IMRESH_LIBS} "tests" "benchmark" )

    enable_testing()
    add_test(NAME mainTestAll COMMAND mainTestAll)

    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS mainTestAll)
endif()
