cmake_minimum_required(VERSION 2.8.5)
project(imresh)

# Options
option(RUN_TESTS OFF)
option(BUILD_EXAMPLES OFF)
option(IMRESH_DEBUG OFF)

# General definitions
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Finding packages, calling FindX.cmake and so on
find_package(CUDA REQUIRED)
find_package(Splash REQUIRED)
find_package(OpenMP REQUIRED)
find_package(FFTW REQUIRED)

if(IMRESH_DEBUG)
    list(APPEND CMAKE_CXX_FLAGS "-g -std=c++11 -fPIC ${OpenMP_CXX_FLAGS}")
else(IMRESH_DEBUG)
    list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -fPIC ${OpenMP_CXX_FLAGS}")
endif(IMRESH_DEBUG)

if(BUILD_EXAMPLES)
    file(GLOB_RECURSE EXAMPLES_SOURCE_FILES ${PROJECT_SOURCE_DIR}/examples/*.cpp ${PROJECT_SOURCE_DIR}/examples/*.hpp)
    add_executable("examples" ${EXAMPLES_SOURCE_FILES})
    target_link_libraries("examples" ${PROJECT_NAME})
endif(BUILD_EXAMPLES)

file(GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.hpp ${PROJECT_SOURCE_DIR}/src/*.cu ${PROJECT_SOURCE_DIR}/src/*.h)
include_directories(${PROJECT_SOURCE_DIR}/src/imresh SYSTEM ${CUDA_INCLUDE_DIRS} ${Splash_INCLUDE_DIRS} ${OpenMP_INCLUDE_DIRS} ${FFTW_INCLUDES})
add_definitions(${Splash_DEFINITIONS})
cuda_include_directories(${PROJECT_SOURCE_DIR}/src/imresh)
cuda_add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${Splash_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${OpenMP_LIBRARIES} ${FFTW_LIBRARIES})
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
