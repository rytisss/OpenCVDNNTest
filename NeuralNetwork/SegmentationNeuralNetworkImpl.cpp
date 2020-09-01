#include "SegmentationNeuralNetworkImpl.h"
#include <filesystem>

#define NORMALIZATION_CONST 0.0039215686

SegmentationNeuralNetwork::SegmentationNeuralNetworkImpl::SegmentationNeuralNetworkImpl() :
	m_modelPath("")
	, m_intialized(false)
	, m_inputChannelsCount(0)
	, m_segmentation_net(nullptr)
	, m_accelerationTarget(cv::dnn::Target::DNN_TARGET_OPENCL)
	, m_backend(cv::dnn::Backend::DNN_BACKEND_DEFAULT)
{
}

bool SegmentationNeuralNetwork::SegmentationNeuralNetworkImpl::Init(const std::string modelPath)
{
	m_accelerationTarget = cv::dnn::Target::DNN_TARGET_OPENCL;
	m_backend = cv::dnn::Backend::DNN_BACKEND_DEFAULT;
	//check the acceletation back, if cuda available and OpenCV build is with cuda, use it
	bool compiledWithCuda = false;
	if (compiledWithCuda)
	{
		std::cout << "OpenCV compiled with CUDA!" << std::endl;
		int cudaDevices = 0;
		if (cudaDevices > 0)
		{
			m_accelerationTarget = cv::dnn::Target::DNN_TARGET_CUDA;
			m_backend = cv::dnn::Backend::DNN_BACKEND_CUDA;
			std::cout << "Backend switch to CUDA!" << std::endl;
		}
		else
		{
			std::cout << "No CUDA devices, acceleration will stay OpenCL!" << std::endl;
		}
	}
	else
	{
		std::cout << "OpenCV is not compiled with CUDA!" << std::endl;
	}

	bool result = false;
	if (!m_intialized)
	{
		cv::Size size;
		int channels;
		if (!GetNeuralNetworkInputInfo(modelPath, size, channels))
		{
			return false;
		}
		cv::dnn::Net neuralNet = cv::dnn::readNetFromTensorflow(modelPath);
		neuralNet.setPreferableBackend(m_backend);
		neuralNet.setPreferableTarget(m_accelerationTarget);

		if (neuralNet.empty())
		{
			std::cout << "Can't load network!" << std::endl;
		}
		else
		{
			if (m_segmentation_net != nullptr)
			{
				delete m_segmentation_net;
			}
			m_segmentation_net = new cv::dnn::Net(neuralNet);
			m_nnSize = size;
			m_inputChannelsCount = channels;
			m_modelPath = modelPath;
			m_intialized = true;
			std::cout << "Network initialized!" << std::endl;
			result = true;
		}
	}
	else
	{
		std::cout << "Network is already initialized!" << std::endl;
	}
	return result;
}

bool SegmentationNeuralNetwork::SegmentationNeuralNetworkImpl::IsInitialized()
{
	return m_intialized;
}

cv::Size SegmentationNeuralNetwork::SegmentationNeuralNetworkImpl::GetNetworkInputSize()
{
	return m_nnSize;
}

int SegmentationNeuralNetwork::SegmentationNeuralNetworkImpl::GetInputChannelsCount()
{
	return (int)m_inputChannelsCount;
}

cv::Mat SegmentationNeuralNetwork::SegmentationNeuralNetworkImpl::Predict(cv::Mat& inputImage)
{
	return Predict(inputImage, NORMALIZATION_CONST, false);
}

cv::Mat SegmentationNeuralNetwork::SegmentationNeuralNetworkImpl::Predict(cv::Mat& inputImage, double scaleFactor, bool swapRB)
{
	cv::Mat output;
	if (m_intialized)
	{
		cv::Mat blob;
		cv::dnn::blobFromImage(inputImage, blob, scaleFactor, m_nnSize, cv::Scalar(0, 0, 0), swapRB, false);
		m_segmentation_net->setInput(blob);
		cv::Mat prob = m_segmentation_net->forward();
		std::vector<cv::Mat> imagesFloat32;
		cv::dnn::imagesFromBlob(prob, imagesFloat32);
		if (imagesFloat32.size() > 0)
		{
			cv::Mat probImage = imagesFloat32[0] * 255.f;
			probImage.convertTo(output, CV_8UC1);
		}
		else
		{
			std::cout << "Zero images received from neural network prediction!" << std::endl;
		}
	}
	else
	{
		std::cout << "Can't predict, network is not initialized!" << std::endl;
	}
	return output;
}

void  SegmentationNeuralNetwork::SegmentationNeuralNetworkImpl::Deinit()
{
	if (m_intialized)
	{
		if (m_segmentation_net != nullptr)
		{
			delete m_segmentation_net;
			m_segmentation_net = nullptr;
			std::cout << "Neural network pointer released!" << std::endl;
		}
		m_nnSize = cv::Size(0, 0);
		m_modelPath = "";
		m_intialized = false;
		m_inputChannelsCount = 0;
	}
}

SegmentationNeuralNetwork::SegmentationNeuralNetworkImpl::~SegmentationNeuralNetworkImpl()
{
	Deinit();
}

bool SegmentationNeuralNetwork::SegmentationNeuralNetworkImpl::GetNeuralNetworkInputInfo(std::string neuralNetPath, cv::Size& size, int& channels)
{
	//read helper file to get neural network input size
	//File format:
	//{
	//	"name": "Hole Detection",
	//	"inputWidth" : 480,
	//	"inputHeight" : 480,
	//	"channels" : 1
	//}
	std::string path = std::filesystem::path(neuralNetPath).parent_path().string();
	std::string fileName = std::filesystem::path(neuralNetPath).filename().stem().string();
	//Helper file with neural network dimensions and other information is in .json file with same name in same directory
	std::string helperFilePath = path + "//" + fileName + ".json";
	if (!std::filesystem::exists(std::filesystem::path(helperFilePath)))
	{
		std::cout << "Neural network " << neuralNetPath << " helper file " << helperFilePath << " doesn't exist! Network won't be deployed!" << std::endl;
		return false;
	}
	cv::FileStorage helperFile;
	helperFile.open(helperFilePath, cv::FileStorage::READ);
	if (!helperFile.isOpened())
	{
		std::cout << "Can't open helper file " << helperFilePath << std::endl;
	}
	std::string modelName_ = "";
	int inputWidth_ = -1;
	int inputHeight_ = -1;
	int channels_ = -1;
	try
	{
		modelName_ = helperFile["name"];
		inputWidth_ = helperFile["inputWidth"];
		inputHeight_ = helperFile["inputHeight"];
		channels_ = helperFile["channels"];
	}
	catch (std::exception& ex)
	{
		std::cout << "Corrupted helper file!" << std::endl;
		PrintFileHelper(helperFilePath);
		std::cout << ex.what() << std::endl;
		helperFile.release();
		return false;
	}
	size = cv::Size(inputWidth_, inputHeight_);
	channels = channels_;
	helperFile.release();
	return true;
}

void SegmentationNeuralNetwork::SegmentationNeuralNetworkImpl::PrintFileHelper(std::string helperFilePath)
{
	std::cout << "Check if the helper file " << helperFilePath << " is such as the following:" << std::endl;
	std::string helperStructure = "{\n"
		"\"name\": \"Hole Detection\", \n"
		"\"inputWidth\" : 480, \n"
		"\"inputHeight\" : 480, \n"
		"\"channels\" : 1\n}";
	std::cout << helperStructure << std::endl;
}

