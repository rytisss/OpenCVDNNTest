cmake_minimum_required(VERSION 3.17.0)

project(NeuralNetwork)
set(CMAKE_BUILD_PARALLEL_LEVEL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# opencv
file(TO_CMAKE_PATH ../../opencv_build/install/ opencv_path)
set(OpenCV_DIR ${opencv_path})
find_package(OpenCV COMPONENTS core imgproc dnn)

add_definitions( -DNEURALNETWORK_EXPORT )

include_directories(
	${OpenCV_INCLUDE_DIRS}
)

add_library(NeuralNetwork SHARED 
	SegmentationNeuralNetwork.h
	SegmentationNeuralNetwork.cpp
	SegmentationNeuralNetworkImpl.h
	SegmentationNeuralNetworkImpl.cpp
)

target_link_libraries(NeuralNetwork PRIVATE
	${OpenCV_LIBS}
)