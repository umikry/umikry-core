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

#ifndef UMIKRY_DETECTOR_HPP
#define UMIKRY_DETECTOR_HPP

#include <vector>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

enum class DetectionMethod { HAAR, CAFFE };

class UmikryFaceDetector {
	private:
		static const std::string CaffeModelUrl;
		static const std::string CaffeConfigUrl;
		static const std::string PathToCaffeModel;
		static const std::string PathToCaffeConfig;
		static const int DownloadChunkSize = 1024;

		static bool downloadUrlAndSaveContent(const std::string& url, const std::string& targetPath);
		static bool fileExists(const std::string& path);

    public:
        UmikryFaceDetector(DetectionMethod detection_method, const std::string& model_path);
        ~UmikryFaceDetector() {};

        std::vector<cv::Rect> detect(const cv::Mat& image);

    private:
        std::vector<cv::Rect> haar_detection(const cv::Mat& image);
        std::vector<cv::Rect> caffe_detection(const cv::Mat& image);

        DetectionMethod detection_method;
        std::string model_path;
};

#endif