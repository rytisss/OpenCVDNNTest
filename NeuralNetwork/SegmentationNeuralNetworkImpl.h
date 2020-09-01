#ifndef SEGMENTATIONNEURALNETWORKIMPL_H
#define SEGMENTATIONNEURALNETWORKIMPL_H

#include "SegmentationNeuralNetwork.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"
#include <iostream>
#include <atomic>
#include <memory>

class SegmentationNeuralNetwork::SegmentationNeuralNetworkImpl
{
public:
	SegmentationNeuralNetworkImpl();
	//Initialize network
	//Model size needs to be known
	bool Init(const std::string modelPath);
	bool IsInitialized();
	cv::Size GetNetworkInputSize();
	int GetInputChannelsCount();
	cv::Mat Predict(cv::Mat& inputImage);
	cv::Mat Predict(cv::Mat& inputImage, double scaleFactor, bool swapRB = false);
	void Deinit();
	~SegmentationNeuralNetworkImpl();
private:
	//internal acceleration target flag
	cv::dnn::Target m_accelerationTarget;
	//internal backend flag
	cv::dnn::Backend m_backend;
	//neural network initialization flag
	std::atomic_bool m_intialized;
	//neural network model path
	std::string m_modelPath;
	//neural network input size
	cv::Size m_nnSize;
	//neural network input image channels count
	size_t m_inputChannelsCount;
	//Get neural network input size, returns false if helper file doesn't exist
	bool GetNeuralNetworkInputInfo(std::string neuralNetPath, cv::Size& size, int& channels);
	//Print json file structure with neural network parameters
	void PrintFileHelper(std::string helperFilePath);
	cv::dnn::Net* m_segmentation_net;
};

#endif //SEGMENTATIONNEURALNETWORKIMPL_H

