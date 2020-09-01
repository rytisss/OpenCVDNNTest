#include "SegmentationNeuralNetwork.h"
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"

int main()
{
	SegmentationNeuralNetwork network;
	std::string modelPath = "C:/src/RoadPavementSegmentation/models/frozen_models/UNet4_res_assp_5x5_24k_128x128_e09_leaky_as_relu.pb";
	modelPath = "C:/src/RoadPavementSegmentation/models/frozen_models/UNet4_res_assp_5x5_16k_320x320_e14.pb";
	network.Init(modelPath);
	std::string imagePath = "C:/Users/prorega/Desktop/fredaTest/hole.jpg";
	cv::Mat image = cv::imread(imagePath, cv::ImreadModes::IMREAD_GRAYSCALE);
	cv::Mat prediction = network.Predict(image);
	int h = 4;
}