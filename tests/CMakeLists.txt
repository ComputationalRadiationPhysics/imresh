
set(CMAKE_CXX_FLAGS "-O1 -g -std=c++11 -fPIC ${OpenMP_CXX_FLAGS}")


if(RUN_TESTS)
    message("Building tests, srcdir=${PROJECT_SOURCE_DIR}, flags=${CMAKE_CXX_FLAGS}")

    add_executable("testVectorIndex" ${PROJECT_SOURCE_DIR}/tests/testVectorIndex.cpp)
    target_link_libraries("testVectorIndex" ${PROJECT_NAME})

    add_executable("testVectorReduce" ${PROJECT_SOURCE_DIR}/tests/testVectorReduce.cpp)
    target_link_libraries("testVectorReduce" ${PROJECT_NAME})

    enable_testing()
    add_test(NAME testVectorIndex COMMAND testVectorIndex)
    add_test(NAME testVectorReduce COMMAND testVectorReduce)
endif()
