#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <vector>
#include <map>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

enum class DetectionMethod { HAAR, CAFFE };

class UmikryFaceDetector {
    public:
        UmikryFaceDetector(DetectionMethod detection_method, std::string model_path);
        ~UmikryFaceDetector() {};

        std::vector<cv::Rect> detect(cv::Mat image);

    private:
        std::vector<cv::Rect> haar_detection(cv::Mat image);
        std::vector<cv::Rect> caffe_detection(cv::Mat image);

        DetectionMethod detection_method;
        std::string model_path;
};

#endif