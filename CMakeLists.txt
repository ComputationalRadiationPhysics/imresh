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
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
if(USE_FFTW)
    find_package(FFTW REQUIRED)
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
                           DEPENDS ${DOXYGEN_INPUT})
        add_custom_target(doc ALL DEPENDS ${DOXYGEN_OUTPUT})
        install(DIRECTORY ${DOXYGEN_OUTPUT} DESTINATION share/doc)
    endif()
endif()

if(IMRESH_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -Wno-unused-parameter -O2 -g -std=c++11 -fPIC -pthread ${OpenMP_CXX_FLAGS}")
    add_definitions("-DIMRESH_DEBUG")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -Wall,-Wextra,-Wshadow -G -lineinfo -Xptxas=-v")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -Wno-unused-parameter -O2 -DNDEBUG -std=c++11 -fPIC -pthread ${OpenMP_CXX_FLAGS}")
endif()
# sm_30 is needed for __shfl_down
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30")

if(BUILD_EXAMPLES)
    file( GLOB_RECURSE EXAMPLES_LIB_FILES ${PROJECT_SOURCE_DIR}/examples/createTestData/*.cpp ${PROJECT_SOURCE_DIR}/examples/createTestData/*.hpp )
    add_library( examples STATIC ${EXAMPLES_LIB_FILES} )
    target_link_libraries( examples )

    add_executable( "threadedExample" ${PROJECT_SOURCE_DIR}/examples/threadedExample.cpp )
    target_link_libraries( "threadedExample" ${PROJECT_NAME} examples )

    add_executable( "miniExample" ${PROJECT_SOURCE_DIR}/examples/miniExample.cpp )
    target_link_libraries( "miniExample" ${PROJECT_NAME} examples )

    add_executable( "outputExampleCreation" ${PROJECT_SOURCE_DIR}/examples/outputExampleCreation.cpp )
    target_link_libraries( "outputExampleCreation" ${PROJECT_NAME} examples )
endif()

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

file(GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.hpp ${PROJECT_SOURCE_DIR}/src/*.cu ${PROJECT_SOURCE_DIR}/src/*.h)
include_directories(${PROJECT_SOURCE_DIR}/src/imresh SYSTEM ${CUDA_INCLUDE_DIRS} ${PNGwriter_INCLUDE_DIRS} ${Splash_INCLUDE_DIRS} ${OpenMP_INCLUDE_DIRS})
cuda_include_directories(${PROJECT_SOURCE_DIR}/src/imresh)
cuda_add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
set( LIBRARIES_NEEDED ${LIBRARIES_NEEDED} ${PNGwriter_LIBRARIES} ${Splash_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${OpenMP_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} )
target_link_libraries( ${PROJECT_NAME} ${LIBRARIES_NEEDED} )
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

# Tests and Benchmarks
set(CMAKE_CXX_FLAGS "-O2 -Wall -Wextra -Wno-unused-parameter -g -std=c++11 ${OpenMP_CXX_FLAGS}")
include_directories( ${PROJECT_SOURCE_DIR}/tests )
if(RUN_TESTS)
    file( GLOB_RECURSE TEST_SOURCE_FILES ${PROJECT_SOURCE_DIR}/tests/*.cpp ${PROJECT_SOURCE_DIR}/tests/*.hpp )
    add_library("tests" ${TEST_SOURCE_FILES})
    target_link_libraries("tests")

    add_executable("testVectorIndex" ${PROJECT_SOURCE_DIR}/tests/imresh/libs/testVectorIndex.cpp)
    target_link_libraries("testVectorIndex" ${PROJECT_NAME} "tests")

    add_executable("testVectorReduce" ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/testVectorReduce.cpp)
    target_link_libraries("testVectorReduce" ${PROJECT_NAME} "tests")

    add_executable("testGaussian" ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/testGaussian.cpp)
    target_link_libraries("testGaussian" ${PROJECT_NAME} "tests")

    enable_testing()
    add_test(NAME testVectorIndex COMMAND testVectorIndex)
    add_test(NAME testVectorReduce COMMAND testVectorReduce)

    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS testVectorIndex testVectorReduce)

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")

    add_executable("profileVectorReduce" ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/profileVectorReduce.cpp)
    target_link_libraries("profileVectorReduce" ${PROJECT_NAME} )

    add_executable("profileGaussian" ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/profileGaussian.cpp)
    target_link_libraries("profileGaussian" ${PROJECT_NAME} )
endif()
