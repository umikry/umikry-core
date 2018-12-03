/*
Copyright (c) 2018, umikry.com
License AGPL-3.0

This code is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License, version 3,
as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License, version 3,
along with this program.  If not, see <http://www.gnu.org/licenses/>
*/

#include "FaceDetector.hpp"

using namespace umikry;

const std::string FaceDetector::PathToCaffeModel = "res10_300x300_ssd_iter_140000_fp16.caffemodel";
const std::string FaceDetector::PathToCaffeConfig = "deploy.prototxt";
const std::string FaceDetector::CaffeModelUrl = "https://github.com/opencv/opencv_3rdparty/raw/19512576c112aa2c7b6328cb0e8d589a4a90a26d/res10_300x300_ssd_iter_140000_fp16.caffemodel";
const std::string FaceDetector::CaffeConfigUrl = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt";
const double FaceDetector::CaffeDetectionConfidence = 0.4;

#ifdef WIN32
#pragma comment(lib, "urlmon.lib")
#include <urlmon.h>
#else // WIN32
#include <stdio.h>
#include <curl/curl.h>

size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
	return fwrite(ptr, size, nmemb, stream);
}
#endif

#include <fstream>
bool FaceDetector::downloadUrlAndSaveContent(const std::string & url, const std::string& targetPath) {
#ifdef WIN32
	IStream* stream;
	auto result = URLOpenBlockingStream(0, url.c_str(), &stream, 0, 0);
	if (result != 0) {
		return false;
	}
	char buffer[DownloadChunkSize];
	unsigned long bytesRead;
	std::ofstream output(targetPath, std::ios_base::binary | std::ios_base::out);
	stream->Read(buffer, DownloadChunkSize, &bytesRead);
	while (bytesRead > 0U) {
		output.write(buffer, (long long)bytesRead);
		stream->Read(buffer, DownloadChunkSize, &bytesRead);
	}
	stream->Release();
	output.close();
	return true;
#else // WIN32
	CURL *curl;
	FILE *fp;
	CURLcode res;
	curl = curl_easy_init();
	if (curl) {
		fp = fopen(targetPath.c_str(), "wb");
		curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
		res = curl_easy_perform(curl);
		curl_easy_cleanup(curl);
		fclose(fp);
		return true;
	}
	return false;
#endif // WIN32
}

bool FaceDetector::fileExists(const std::string& path) {
	std::ifstream f(path.c_str());
	return f.good();
}

FaceDetector::FaceDetector(DetectionMethod detectionMethod, const std::string& modelPath)
	: detectionMethod(detectionMethod)
	, modelPath(modelPath)
	, caffeDetectorNet(nullptr) {
	
	switch (detectionMethod) {
	case DetectionMethod::HAAR:
		initHaar();
		break;
	case DetectionMethod::CAFFE:
		initCaffe();
		break;
	default:
		throw std::runtime_error("Invalid or not supported model specified.");
	}
}

FaceDetector::~FaceDetector() {
	std::lock_guard<std::mutex> guard(caffeMutex);
	if (caffeDetectorNet != nullptr) {
		delete caffeDetectorNet;
	}
}

void FaceDetector::initCaffe() {
	if (!fileExists(caffeWeightFilePath())) {
		if (!downloadUrlAndSaveContent(CaffeModelUrl, caffeWeightFilePath())) {
			throw std::runtime_error("Could not download Caffe model weights.");
		}
	}

	if (!fileExists(caffeConfigFilePath())) {
		if (!downloadUrlAndSaveContent(CaffeConfigUrl, caffeConfigFilePath())) {
			throw std::runtime_error("Could not download Caffe config.");
		}
	}

	std::lock_guard<std::mutex> guard(caffeMutex);
	caffeDetectorNet = new cv::dnn::Net(cv::dnn::readNetFromCaffe(caffeConfigFilePath(), caffeWeightFilePath()));
}

void FaceDetector::initHaar() {

}

std::vector<cv::Rect> FaceDetector::detect(const cv::Mat& image) {
	switch (detectionMethod) {
	case DetectionMethod::HAAR:
		return haarDetection(image);
	case DetectionMethod::CAFFE:
		return caffeDetection(image);
	default:
		throw std::runtime_error("Invalid or not supported model specified.");
		return {};
	}
}

std::vector<cv::Rect> FaceDetector::haarDetection(const cv::Mat& image) {
	std::cout << "The method haar_detect() is not implemented yet." << std::endl;
    return {};
}

std::vector<cv::Rect> FaceDetector::caffeDetection(const cv::Mat& image) {
	caffeMutex.lock();
	caffeDetectorNet->setInput(cv::dnn::blobFromImage(image, 0.4, image.size(), cv::Scalar(104, 117, 123), false, false), "data");
    cv::Mat detection = caffeDetectorNet->forward("detection_out");
	caffeMutex.unlock();

    const cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    std::vector<cv::Rect> boundingBoxes = {};
    for(int i = 0; i < detectionMat.rows; ++i) {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > CaffeDetectionConfidence) {
            const int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
			const int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
			const int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
			const int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);

            const cv::Rect boundingBox(x1, y1, x2 - x1, y2 - y1);
			boundingBoxes.push_back(boundingBox);
        }
    }
    return boundingBoxes;
}
