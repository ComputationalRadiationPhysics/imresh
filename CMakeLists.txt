cmake_minimum_required(VERSION 3.3.1)
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

# find HaLT
find_path(
    _HALT_ROOT_DIR
    NAMES "include/haLT/FFT.hpp"
    HINTS ENV HALT_ROOT
    DOC "HaLT location"
)
set( _HALT_ROOT_DIR "${PROJECT_SOURCE_DIR}/halt" )
if( NOT EXISTS "${_HALT_ROOT_DIR}/include/haLT/FFT.hpp" )
    message(FATAL_ERROR "HaLT could not be found! Please set $HALT_ROOT to the correct directory.")
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

if(BUILD_DOC)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_INPUT_PRE ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
        set(DOXYGEN_INPUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
        set(DOXYGEN_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html)
        set(DOXY_MAIN_PAGE ${PROJECT_SOURCE_DIR}/README.md)
        configure_file(${DOXYGEN_INPUT_PRE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
        add_custom_command(OUTPUT ${DOXYGEN_OUTPUT}
                           COMMAND ${CMAKE_COMMAND} -E echo_append "Building API Documentation..."
                           COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
                           COMMAND ${CMAKE_COMMAND} -E echo "Done."
                           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                           DEPENDS ${DOXYGEN_INPUT} ${DOXY_MAIN_PAGE})
        add_custom_target(doc ALL DEPENDS ${DOXYGEN_OUTPUT})
        install(DIRECTORY ${DOXYGEN_OUTPUT} DESTINATION share/doc)
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

if(USE_PNG)
    # if PNGWRITER_ROOT not set, then get and compile submodule
    if( "$ENV{PNGWRITER_ROOT}" STREQUAL "" )
        set( PNGWRITER_ROOT "${PROJECT_SOURCE_DIR}/pngwriter" )
        if( NOT EXISTS "${PNGWRITER_ROOT}/include/pngwriter.h" OR
            NOT EXISTS "${PNGWRITER_ROOT}/lib/libpngwriter.a"  OR
            NOT EXISTS "${PNGWRITER_ROOT}/lib/libpngwriter.so"
        )
            # every command own execute process because of this bug:
            execute_process(
                COMMAND sh -c "\
                git submodule update --init -- ./pngwriter && \
                mkdir -p ./pngwriter/build                 && \
                cd ./pngwriter/build                       && \
                cmake .. -DCMAKE_INSTALL_PREFIX=..         && \
                make install -j 4                             \
                "
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            )
        endif()
        set( ENV{PNGWRITER_ROOT} "${PNGWRITER_ROOT}")
    endif()

    find_package( PNGwriter REQUIRED )
    add_definitions( "-DUSE_PNG" ${PNGwriter_DEFINITIONS} )
endif()

if(USE_TIFF)
    find_package(TIFF REQUIRED)
    add_definitions( "-DUSE_TIFF" )
endif()

if(USE_SPLASH)
    # if PNGWRITER_ROOT not set, then get and compile submodule
    if( "$ENV{SPLASH_ROOT}" STREQUAL "" )
        set( SPLASH_ROOT "${PROJECT_SOURCE_DIR}/libsplash" )
        if( NOT EXISTS "${SPLASH_ROOT}/include/splash/splash.h" OR
            NOT EXISTS "${SPLASH_ROOT}/lib/libsplash.a" OR
            NOT EXISTS "${SPLASH_ROOT}/lib/libsplash.so"
        )
            # every command own execute process because of this bug:
            execute_process(
                COMMAND sh -c "
                git submodule update --init -- ./libsplash && \
                mkdir -p  ./libsplash/build                && \
                cd ./libsplash/build                       && \
                cmake .. -DCMAKE_INSTALL_PREFIX=..         && \
                make install -j 4                             \
                "
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            )
        endif()
        set( ENV{SPLASH_ROOT} "${SPLASH_ROOT}")
    endif()

    find_package(Splash REQUIRED)
    add_definitions("-DUSE_SPLASH" ${Splash_DEFINITIONS})
endif()

if(USE_FFTW)
    include_directories( ${FFTW_INCLUDES} )
    set( LIBRARIES_NEEDED ${LIBRARIES_NEEDED} ${FFTW_LIBRARIES} )
    add_definitions("-DUSE_FFTW" ${FFTW_DEFINITIONS})
endif()

# add files and compile
set( SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/imresh/algorithms/shrinkWrap.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/algorithms/vectorElementwise.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/algorithms/vectorReduce.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/algorithms/cuda/cudaGaussian.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/algorithms/cuda/cudaShrinkWrap.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/algorithms/cuda/cudaVectorElementwise.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/algorithms/cuda/cudaVectorReduce.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/libs/calcGaussianKernel.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/libs/cudacommon.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/libs/CudaKernelConfig.cpp
    # makes use of cufft and only cufft, therefore could be a nice test example for a cufft_to_cupla.hpp test
    ${PROJECT_SOURCE_DIR}/src/imresh/libs/diffractionIntensity.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/libs/fftShift.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/libs/gaussian.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/libs/hybridInputOutput.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/libs/vectorIndex.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/io/ReadTiff.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/io/taskQueue.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/io/readInFuncs/readInFuncs.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/io/writeOutFuncs/writeOutFuncs.cpp
)
include_directories(
    ${PROJECT_SOURCE_DIR}/src/imresh
)
include_directories(
    SYSTEM
    ${CUDA_INCLUDE_DIRS}
    ${PNGwriter_INCLUDE_DIRS}
    ${Splash_INCLUDE_DIRS}
    ${OpenMP_INCLUDE_DIRS}
    ${cupla_INCLUDE_DIRS}
    ${TIFF_INCLUDE_DIRS}
    ${HaLT_INCLUDE_DIRS}
)
message( STATUS "CUDA_INCLUDE_DIRS      = ${CUDA_INCLUDE_DIRS}"      )
message( STATUS "PNGwriter_INCLUDE_DIRS = ${PNGwriter_INCLUDE_DIRS}" )
message( STATUS "Splash_INCLUDE_DIRS    = ${Splash_INCLUDE_DIRS}"    )
message( STATUS "OpenMP_INCLUDE_DIRS    = ${OpenMP_INCLUDE_DIRS}"    )
message( STATUS "cupla_INCLUDE_DIRS     = ${cupla_INCLUDE_DIRS}"     )
message( STATUS "TIFF_INCLUDE_DIRS      = ${TIFF_INCLUDE_DIRS}"      )
message( STATUS "HaLT_INCLUDE_DIRS      = ${HaLT_INCLUDE_DIRS}"      )

#cuda_include_directories(${PROJECT_SOURCE_DIR}/src/imresh)
#cuda_add_library(${PROJECT_NAME} ${SOURCE_FILES} STATIC)
# Always add all files to the target executable build call to add them to the build project.
alpaka_add_library(
    ${PROJECT_NAME}
    ${SOURCE_FILES}
    ${cupla_SOURCE_FILES}
)
set( LIBRARIES_NEEDED ${LIBRARIES_NEEDED}
    # abstraction layers
    ${_cupla_LINK_LIBRARIES_PUBLIC}
    ${HaLT_LIBRARIES}
    # runtime libraries (note that normal only either FFTW_LIBRARIES or CUDA_CUFFT_LIBRARIES needs to be set
    ${FFTW_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
    ${CUDA_LIBRARIES}
    # image reading and writing libraries
    ${Splash_LIBRARIES}
    ${PNGwriter_LIBRARIES}
    ${TIFF_LIBRARIES}
    # OS relevant libraries
    ${OpenMP_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)
message( "{FFTW_LIBRARIES} = ${FFTW_LIBRARIES}" )
target_link_libraries( ${PROJECT_NAME} ${LIBRARIES_NEEDED}  )
set( IMRESH_LIBS ${PROJECT_NAME} ${LIBRARIES_NEEDED} )
#install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

if(BUILD_EXAMPLES OR RUN_TESTS)
    file( GLOB_RECURSE EXAMPLES_LIB_FILES ${PROJECT_SOURCE_DIR}/examples/createTestData/*.cpp ${PROJECT_SOURCE_DIR}/examples/createTestData/*.hpp )
    alpaka_add_library( examples STATIC ${EXAMPLES_LIB_FILES} )
    target_link_libraries( examples )
endif()
if(BUILD_EXAMPLES)
    alpaka_add_executable( "threadedExample" ${PROJECT_SOURCE_DIR}/examples/threadedExample.cpp )
    target_link_libraries( "threadedExample" examples ${IMRESH_LIBS} )

    alpaka_add_executable( "miniExample" ${PROJECT_SOURCE_DIR}/examples/miniExample.cpp )
    target_link_libraries( "miniExample" examples ${IMRESH_LIBS} )

    alpaka_add_executable( "outputExampleCreation" ${PROJECT_SOURCE_DIR}/examples/outputExampleCreation.cpp )
    target_link_libraries( "outputExampleCreation" examples ${IMRESH_LIBS} )
endif()

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
        ${PROJECT_SOURCE_DIR}/tests/benchmarkHelper.hpp
        ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/testVectorElementwise.hpp
        ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/testVectorElementwise.cpp
        ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/testVectorReduce.hpp
        ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/testVectorReduce.cpp
        ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/testShrinkWrap.cpp
        ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/testShrinkWrap.hpp
        ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/testGaussian.cpp
        ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/testGaussian.hpp
        ${PROJECT_SOURCE_DIR}/tests/imresh/libs/testVectorIndex.cpp
        ${PROJECT_SOURCE_DIR}/tests/imresh/libs/testVectorIndex.hpp
        ${PROJECT_SOURCE_DIR}/tests/imresh/io/testReadWrite.cpp
        ${PROJECT_SOURCE_DIR}/tests/imresh/io/testReadWrite.hpp
        ${PROJECT_SOURCE_DIR}/tests/mainTestAll.cpp
    )
    alpaka_add_library( "tests" ${TEST_SOURCE_FILES} STATIC )
    target_link_libraries("tests" ${IMRESH_LIBS} "benchmark")

    alpaka_add_executable("mainTestAll" ${PROJECT_SOURCE_DIR}/tests/mainTestAll.cpp )
    target_link_libraries("mainTestAll" ${IMRESH_LIBS} "tests" "benchmark" "examples" )

    enable_testing()
    add_test(NAME mainTestAll COMMAND mainTestAll)

    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS mainTestAll)

    # the following are very small executable units for the use with nvvp as too complex programs are difficult to profile (threadMainExample makes nvvp use 6+GB RAM)
    #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")
    #
    #add_executable("profileVectorReduce" ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/profileVectorReduce.cpp)
    #target_link_libraries("profileVectorReduce" ${IMRESH_LIBS} )
    #
    #add_executable("profileGaussian" ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/profileGaussian.cpp)
    #target_link_libraries("profileGaussian" ${IMRESH_LIBS} )
endif()
