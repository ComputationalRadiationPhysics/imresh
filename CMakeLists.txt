cmake_minimum_required(VERSION 3.4.1)
project(imresh)

# Options
option(RUN_TESTS      "Runs some unit tests including some benchmarks" OFF)
option(BUILD_EXAMPLES "Builds some examples showing how to make use of shrinkWrap(...)" OFF)
option(IMRESH_DEBUG   "Enables debugging code, especially many asserts to check for correctness" OFF)
option(BUILD_DOC      "Builds Doxygen Documentation" ON)
option(USE_PNG        "Enables PNG output of reconstructed image" OFF)
option(USE_SPLASH     "Enables HDF5 input and output of images" OFF)

# General definitions
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Finding packages, calling FindX.cmake and so on
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(FFTW REQUIRED)
find_package(Threads REQUIRED)

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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -g -std=c++11 -fPIC -pthread ${OpenMP_CXX_FLAGS}")
    add_definitions("-DIMRESH_DEBUG")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -std=c++11 -fPIC -pthread ${OpenMP_CXX_FLAGS}")
endif()
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30")

if(BUILD_EXAMPLES)
    file(GLOB_RECURSE EXAMPLES_SOURCE_FILES ${PROJECT_SOURCE_DIR}/examples/*.cpp ${PROJECT_SOURCE_DIR}/examples/*.hpp)
    add_executable("examples" ${EXAMPLES_SOURCE_FILES})
    target_link_libraries("examples" ${PROJECT_NAME})
endif()

if(USE_PNG)
    find_package(PNGwriter REQUIRED)
    add_definitions("-DUSE_PNG ${PNGwriter_DEFINITIONS}")
endif()

if(USE_SPLASH)
    find_package(Splash REQUIRED)
    add_definitions("-DUSE_SPLASH" ${Splash_DEFINITIONS})
endif()

file(GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.hpp ${PROJECT_SOURCE_DIR}/src/*.cu ${PROJECT_SOURCE_DIR}/src/*.h)
include_directories(${PROJECT_SOURCE_DIR}/src/imresh SYSTEM ${CUDA_INCLUDE_DIRS} ${PNGwriter_INCLUDE_DIRS} ${Splash_INCLUDE_DIRS} ${OpenMP_INCLUDE_DIRS} ${FFTW_INCLUDES})
cuda_include_directories(${PROJECT_SOURCE_DIR}/src/imresh)
cuda_add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${PNGwriter_LIBRARIES} ${Splash_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${OpenMP_LIBRARIES} ${FFTW_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

# Tests and Benchmarks
set(CMAKE_CXX_FLAGS "-std=c++11 ${OpenMP_CXX_FLAGS}")
include_directories( ${PROJECT_SOURCE_DIR}/tests )
if(RUN_TESTS)
    file( GLOB_RECURSE TEST_SOURCE_FILES ${PROJECT_SOURCE_DIR}/tests/*.cpp ${PROJECT_SOURCE_DIR}/tests/*.hpp )
    add_library("tests" ${TEST_SOURCE_FILES})
    target_link_libraries("tests")

    add_executable("testVectorIndex" ${PROJECT_SOURCE_DIR}/tests/imresh/libs/testVectorIndex.cpp)
    target_link_libraries("testVectorIndex" ${PROJECT_NAME} "tests")

    add_executable("testVectorReduce" ${PROJECT_SOURCE_DIR}/tests/imresh/algorithms/testVectorReduce.cpp)
    target_link_libraries("testVectorReduce" ${PROJECT_NAME} "tests")

    enable_testing()
    add_test(NAME testVectorIndex COMMAND testVectorIndex)
    add_test(NAME testVectorReduce COMMAND testVectorReduce)

    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS testVectorIndex testVectorReduce)
endif()
