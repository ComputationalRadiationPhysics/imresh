cmake_minimum_required(VERSION 2.8)
project(imresh)

# General definitions
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Finding files
add_subdirectory(${PROJECT_SOURCE_DIR}/src/imresh)

# Finding packages, calling FindX.cmake and so on
find_package(CUDA QUIET REQUIRED)

# Building/Linking
# Maybe we'll need that in the future
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode)

include_directories(${PROJECT_SOURCE_DIR}/src/imresh)
cuda_include_directories(${PROJECT_SOURCE_DIR}/src/imresh)
cuda_add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
