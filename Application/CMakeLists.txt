cmake_minimum_required(VERSION 3.17.0)

project(NeuralNetworkLoaderTest)
set(CMAKE_BUILD_PARALLEL_LEVEL)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# opencv
file(TO_CMAKE_PATH ../../opencv_build/install/ opencv_path)
set(OpenCV_DIR ${opencv_path})
find_package(OpenCV COMPONENTS core videoio)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	../NeuralNetwork/
 )

add_executable(NeuralNetworkLoaderTest 
	main.cpp
)

add_dependencies(NeuralNetworkLoaderTest
	NeuralNetwork
)

target_link_libraries(NeuralNetworkLoaderTest PRIVATE
	${OpenCV_LIBS}
	NeuralNetwork
)

add_custom_command(TARGET NeuralNetworkLoaderTest POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	$<TARGET_FILE_DIR:NeuralNetwork>/NeuralNetwork.dll
	$<TARGET_FILE_DIR:NeuralNetworkLoaderTest>)
#OpenCV
add_custom_command(TARGET NeuralNetworkLoaderTest POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_core440d.dll>$<$<CONFIG:release>:opencv_core440.dll>
	$<TARGET_FILE_DIR:NeuralNetworkLoaderTest>)
add_custom_command(TARGET NeuralNetworkLoaderTest POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_highgui440d.dll>$<$<CONFIG:release>:opencv_highgui440.dll>
	$<TARGET_FILE_DIR:NeuralNetworkLoaderTest>)
add_custom_command(TARGET NeuralNetworkLoaderTest POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_dnn440d.dll>$<$<CONFIG:release>:opencv_dnn440.dll>
	$<TARGET_FILE_DIR:NeuralNetworkLoaderTest>)
add_custom_command(TARGET NeuralNetworkLoaderTest POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_imgproc440d.dll>$<$<CONFIG:release>:opencv_imgproc440.dll>
	$<TARGET_FILE_DIR:NeuralNetworkLoaderTest>)
add_custom_command(TARGET NeuralNetworkLoaderTest POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_imgcodecs440d.dll>$<$<CONFIG:release>:opencv_imgcodecs440.dll>
	$<TARGET_FILE_DIR:NeuralNetworkLoaderTest>)
add_custom_command(TARGET NeuralNetworkLoaderTest POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_videoio440d.dll>$<$<CONFIG:release>:opencv_videoio440.dll>
	$<TARGET_FILE_DIR:NeuralNetworkLoaderTest>)
add_custom_command(TARGET NeuralNetworkLoaderTest POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_video440d.dll>$<$<CONFIG:release>:opencv_video440.dll>
	$<TARGET_FILE_DIR:NeuralNetworkLoaderTest>)
add_custom_command(TARGET NeuralNetworkLoaderTest POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_calib3d440d.dll>$<$<CONFIG:release>:opencv_calib3d440.dll>
	$<TARGET_FILE_DIR:NeuralNetworkLoaderTest>)
add_custom_command(TARGET NeuralNetworkLoaderTest POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_features2d440d.dll>$<$<CONFIG:release>:opencv_features2d440.dll>
	$<TARGET_FILE_DIR:NeuralNetworkLoaderTest>)
add_custom_command(TARGET NeuralNetworkLoaderTest POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_flann440d.dll>$<$<CONFIG:release>:opencv_flann440.dll>
	$<TARGET_FILE_DIR:NeuralNetworkLoaderTest>)