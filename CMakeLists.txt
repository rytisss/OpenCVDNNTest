cmake_minimum_required (VERSION 3.17.0)
project (NeuralNetworkLoaderTest)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=gnu++0x)
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(CMAKE_BUILD_PARALLEL_LEVEL)

add_subdirectory(Application)
add_subdirectory(NeuralNetwork)