#pragma managed(push, off)
#include "SegmentationNeuralNetworkImpl.h"
#pragma managed(pop)


SegmentationNeuralNetwork::SegmentationNeuralNetwork()
{
	pSegmentationNeuralNetworkImpl = new SegmentationNeuralNetworkImpl();
}

bool SegmentationNeuralNetwork::Init(const std::string modelPath)
{
	return pSegmentationNeuralNetworkImpl->Init(modelPath);
}

bool SegmentationNeuralNetwork::IsInitialized()
{
	return pSegmentationNeuralNetworkImpl->IsInitialized();
}

cv::Size SegmentationNeuralNetwork::GetNetworkInputSize()
{
	return pSegmentationNeuralNetworkImpl->GetNetworkInputSize();
}

int SegmentationNeuralNetwork::GetInputChannelsCount()
{
	return pSegmentationNeuralNetworkImpl->GetInputChannelsCount();
}

cv::Mat SegmentationNeuralNetwork::Predict(cv::Mat& inputImage)
{
	return pSegmentationNeuralNetworkImpl->Predict(inputImage);
}

cv::Mat SegmentationNeuralNetwork::Predict(cv::Mat& inputImage, double scaleFactor, bool swapRB)
{
	return pSegmentationNeuralNetworkImpl->Predict(inputImage, scaleFactor, swapRB);
}

void SegmentationNeuralNetwork::Deinit()
{
	pSegmentationNeuralNetworkImpl->Deinit();
}

SegmentationNeuralNetwork::~SegmentationNeuralNetwork()
{
	delete pSegmentationNeuralNetworkImpl;
}
