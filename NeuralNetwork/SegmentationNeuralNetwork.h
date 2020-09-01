#ifndef SEGMENTATIONNEURALNETWORK_H
#define SEGMENTATIONNEURALNETWORK_H

#if defined (_WIN32) 
#pragma warning(disable: 4273) //disable dll linkage warnings for visual studio
#if defined(NEURALNETWORK_EXPORT)
#define SEGMENTATIONNEURALNETWORK_API __declspec(dllexport)
#else
#define SEGMENTATIONNEURALNETWORK_API __declspec(dllimport)
#endif /* SEGMENTATIONNEURALNETWORK_API */
#define sprintf_s sprintf_s
#else /* defined (_WIN32) */
#define SEGMENTATIONNEURALNETWORK_API
#define _sprintf sprintf
#endif

#include "opencv2/core.hpp"

class SEGMENTATIONNEURALNETWORK_API SegmentationNeuralNetwork
{
public:
	SegmentationNeuralNetwork();
	//Initialize network
	//Model size needs to be known
	bool Init(const std::string modelPath);
	bool IsInitialized();
	cv::Size GetNetworkInputSize();
	int GetInputChannelsCount();
	cv::Mat Predict(cv::Mat& inputImage);
	cv::Mat Predict(cv::Mat& inputImage, double scaleFactor, bool swapRB = false);
	void Deinit();
	~SegmentationNeuralNetwork();
private:
	class SegmentationNeuralNetworkImpl;
	SegmentationNeuralNetworkImpl* pSegmentationNeuralNetworkImpl;
};

#endif //SEGMENTATIONNEURALNETWORKIMPL_H


