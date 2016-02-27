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
#find_package(CUDA 7.0 REQUIRED)
#find_package(OpenMP REQUIRED)
#find_package(Threads REQUIRED)
#if(USE_FFTW)
#    find_package(FFTW REQUIRED)
#endif()

# Find cupla
SET( cupla_ROOT "$ENV{CUPLA_ROOT}" CACHE STRING  "The location of the cupla library" FORCE )
if( NOT EXISTS "${cupla_ROOT}" )
    message(FATAL_ERROR "CUPLA_ROOT points to non-existing path!")
endif()
list( APPEND CMAKE_MODULE_PATH "${cupla_ROOT}" )  # add findCupla.cmake
find_package( "cupla" REQUIRED )
add_definitions( ${cupla_DEFINITIONS} )

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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -Wno-unused-parameter -O2 -g -std=c++11 -fPIC -pthread ${OpenMP_CXX_FLAGS}")
    add_definitions("-DIMRESH_DEBUG")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -Wall,-Wextra,-Wshadow -G -lineinfo -Xptxas=-v")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG -std=c++11 -fPIC -pthread ${OpenMP_CXX_FLAGS}")
endif()
# sm_30 is needed for __shfl_down
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30")

if(USE_PNG)
    find_package(PNGwriter REQUIRED)
    add_definitions("-DUSE_PNG ${PNGwriter_DEFINITIONS}")
endif()

if(USE_SPLASH)
    find_package(Splash REQUIRED)
    add_definitions("-DUSE_SPLASH" ${Splash_DEFINITIONS})
endif()

if(USE_FFTW)
    include_directories( ${FFTW_INCLUDES} )
    set( LIBRARIES_NEEDED ${LIBRARIES_NEEDED} ${FFTW_LIBRARIES} )
    add_definitions("-DUSE_FFTW" ${FFTW_DEFINITIONS})
endif()

# add files and compile
file( GLOB_RECURSE SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/imresh/algorithms/alpaka/*.cpp
    ${PROJECT_SOURCE_DIR}/src/imresh/algorithms/alpaka/*.hpp
    ${PROJECT_SOURCE_DIR}/src/imresh/libs/alpaka_T_ACC.hpp
)
message( "SOURCE_FILES = ${SOURCE_FILES}" )
include_directories(
    ${PROJECT_SOURCE_DIR}/src/imresh
    SYSTEM
    ${CUDA_INCLUDE_DIRS}
    ${PNGwriter_INCLUDE_DIRS}
    ${Splash_INCLUDE_DIRS}
    ${OpenMP_INCLUDE_DIRS}
    ${cupla_INCLUDE_DIRS}
)

#cuda_include_directories(${PROJECT_SOURCE_DIR}/src/imresh)
#cuda_add_library(${PROJECT_NAME} ${SOURCE_FILES} STATIC)
# Always add all files to the target executable build call to add them to the build project.
alpaka_add_library(
    ${PROJECT_NAME}
    ${SOURCE_FILES}
    ${cupla_SOURCE_FILES}
)
set( LIBRARIES_NEEDED ${LIBRARIES_NEEDED}
    ${PNGwriter_LIBRARIES}
    ${Splash_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
    ${OpenMP_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)
target_link_libraries( ${PROJECT_NAME} ${LIBRARIES_NEEDED} PUBLIC ${_cupla_LINK_LIBRARIES_PUBLIC} )
#install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

if(BUILD_EXAMPLES OR RUN_TESTS)
    file( GLOB_RECURSE EXAMPLES_LIB_FILES ${PROJECT_SOURCE_DIR}/examples/createTestData/*.cpp ${PROJECT_SOURCE_DIR}/examples/createTestData/*.hpp )
    add_library( examples STATIC ${EXAMPLES_LIB_FILES} )
    target_link_libraries( examples )
endif()
if(BUILD_EXAMPLES)
    add_executable( "threadedExample" ${PROJECT_SOURCE_DIR}/examples/threadedExample.cpp )
    target_link_libraries( "threadedExample" examples ${PROJECT_NAME} )

    add_executable( "miniExample" ${PROJECT_SOURCE_DIR}/examples/miniExample.cpp )
    target_link_libraries( "miniExample" examples ${PROJECT_NAME} )

    add_executable( "outputExampleCreation" ${PROJECT_SOURCE_DIR}/examples/outputExampleCreation.cpp )
    target_link_libraries( "outputExampleCreation" examples ${PROJECT_NAME} )
endif()

# Tests and Benchmarks
include_directories( ${PROJECT_SOURCE_DIR}/tests )
if(RUN_TESTS)
    file( GLOB_RECURSE BENCHMARK_SOURCE_FILES ${PROJECT_SOURCE_DIR}/benchmark/*.cpp  ${PROJECT_SOURCE_DIR}/benchmark/*.hpp ${PROJECT_SOURCE_DIR}/benchmark/*.cu )
    cuda_add_library("benchmark" ${BENCHMARK_SOURCE_FILES} STATIC)
    target_link_libraries("benchmark" ${PROJECT_NAME})

    # reset cxx flags to delete the -DNDEBUG flag, because tests always need to assert
    set(CMAKE_CXX_FLAGS "-O2 -Wall -Wextra -Wno-unused-parameter -g -std=c++11 ${OpenMP_CXX_FLAGS}")
    # for includes of #include "benchmarks/...". Note that
    #    clude_directories(${PROJECT_SOURCE_DIR})
    # would lead to ambiguous includes
    include_directories(${PROJECT_SOURCE_DIR})
    cuda_include_directories(${PROJECT_SOURCE_DIR})

    file( GLOB_RECURSE TEST_SOURCE_FILES ${PROJECT_SOURCE_DIR}/tests/*.cpp ${PROJECT_SOURCE_DIR}/tests/*.hpp ${PROJECT_SOURCE_DIR}/tests/*.cu )
    cuda_add_library("tests" ${TEST_SOURCE_FILES} STATIC)
    target_link_libraries("tests" ${PROJECT_NAME} "benchmark")

    cuda_add_executable("mainTestAll" ${PROJECT_SOURCE_DIR}/tests/mainTestAll.cpp)
    target_link_libraries("mainTestAll" ${PROJECT_NAME} "tests" "benchmark" "examples")

    enable_testing()
    add_test(NAME mainTestAll COMMAND mainTestAll)

    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS mainTestAll)

    # the following are very small executable units for the use with nvvp as too complex programs are difficult to profile (threadMainExample makes nvvp use 6+GB RAM)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")

    add_executable("profileVectorReduce" ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/profileVectorReduce.cpp)
    target_link_libraries("profileVectorReduce" ${PROJECT_NAME} )

    add_executable("profileGaussian" ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/profileGaussian.cpp)
    target_link_libraries("profileGaussian" ${PROJECT_NAME} )
endif()
