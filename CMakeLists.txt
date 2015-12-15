cmake_minimum_required(VERSION 2.8)
project(imresh)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)


add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
