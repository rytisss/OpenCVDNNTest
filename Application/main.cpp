#include "SegmentationNeuralNetwork.h"
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

int main()
{
	SegmentationNeuralNetwork network;
	std::string modelPath = "C:/src/RoadPavementSegmentation/models/frozen_models/UNet4_res_assp_5x5_24k_128x128_e09_leaky_as_relu.pb";
	modelPath = "C:/src/Projects/OpenCVDNNTest/models/UNet4_res_assp_5x5_24k_128x128_e09_leaky_relu.pb";
	network.Init(modelPath);
	std::string imagePath = "C:/src/Projects/OpenCVDNNTest/images/hole.jpg";
	cv::Mat image = cv::imread(imagePath, cv::ImreadModes::IMREAD_GRAYSCALE);
	try
	{
		cv::Mat prediction = network.Predict(image);
	}
	catch (cv::Exception& ex)
	{
		std::cout << ex.what() << std::endl;
	}
	return 0;
}