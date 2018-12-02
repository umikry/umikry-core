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

#include "UmikryFaceDetector.hpp"

const std::string UmikryFaceDetector::PathToCaffeModel = "res10_300x300_ssd_iter_140000_fp16.caffemodel";
const std::string UmikryFaceDetector::PathToCaffeConfig = "deploy.prototxt";
const std::string UmikryFaceDetector::CaffeModelUrl = "https://github.com/opencv/opencv_3rdparty/raw/19512576c112aa2c7b6328cb0e8d589a4a90a26d/res10_300x300_ssd_iter_140000_fp16.caffemodel";
const std::string UmikryFaceDetector::CaffeConfigUrl = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt";

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
bool UmikryFaceDetector::downloadUrlAndSaveContent(const std::string & url, const std::string& targetPath)
{
#ifdef WIN32
	IStream* stream;
	HRESULT result = URLOpenBlockingStream(0, url.c_str(), &stream, 0, 0);
	if (result != 0)
	{
		return false;
	}
	char buffer[DownloadChunkSize];
	unsigned long bytesRead;
	std::ofstream output(targetPath, std::ios_base::binary | std::ios_base::out);
	stream->Read(buffer, DownloadChunkSize, &bytesRead);
	while (bytesRead > 0U)
	{
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
	if (curl)
	{
		fp = fopen(targetPath.c_str(), "wb");
		curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
		res = curl_easy_perform(curl);
		curl_easy_cleanup(curl);
		fclose(fp);
	}

#endif // WIN32
}

bool UmikryFaceDetector::fileExists(const std::string& path) {
	std::ifstream f(path.c_str());
	return f.good();
}

UmikryFaceDetector::UmikryFaceDetector(DetectionMethod detection_method, const std::string& model_path) {
    this->detection_method = detection_method;
    this->model_path = model_path;
}

std::vector<cv::Rect> UmikryFaceDetector::detect(const cv::Mat& image) {
    if (detection_method == DetectionMethod::HAAR) {
        return haar_detection(image);
    } else if (detection_method == DetectionMethod::CAFFE) {
        return caffe_detection(image);
    } else {
        return {};
    }
}

std::vector<cv::Rect> UmikryFaceDetector::haar_detection(const cv::Mat& image) {
    std::cout << "The method haar_detect() is not implemented yet." << std::endl;
    return {};
}

std::vector<cv::Rect> UmikryFaceDetector::caffe_detection(const cv::Mat& image) {
	const std::string caffeWeightFile = model_path + "/" + PathToCaffeModel;
	if (!fileExists(caffeWeightFile)) {
		if (!downloadUrlAndSaveContent(CaffeModelUrl, caffeWeightFile)) {
			throw std::runtime_error("Could not download Caffe model weights.");
		}
	}

	const std::string caffeConfigFile = model_path + "/" + PathToCaffeConfig;
	if (!fileExists(caffeConfigFile)) {
		if (!downloadUrlAndSaveContent(CaffeConfigUrl, caffeConfigFile)) {
			throw std::runtime_error("Could not download Caffe config.");
		}
	}

	const cv::Mat blob = cv::dnn::blobFromImage(image, 0.4, image.size(), cv::Scalar(104, 117, 123), false, false);
	cv::dnn::Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
    net.setInput(blob, "data");
    cv::Mat detection = net.forward("detection_out");
     
    const cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    
    std::vector<cv::Rect> bounding_boxes = {};
    for(int i = 0; i < detectionMat.rows; ++i) {
        float confidence = detectionMat.at<float>(i, 2);
     
        if(confidence > 0.4) {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);
     
            const cv::Rect bounding_box(x1, y1, x2 - x1, y2 - y1);
            bounding_boxes.push_back(bounding_box);
        }
    }
    return bounding_boxes;
}