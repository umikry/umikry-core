#ifndef UMIKRY_DETECTOR_HPP
#define UMIKRY_DETECTOR_HPP

#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

enum class DetectionMethod { HAAR, CAFFE };

class UmikryFaceDetector {
    public:
        UmikryFaceDetector(DetectionMethod detection_method, const std::string model_path);
        ~UmikryFaceDetector() {};

        std::vector<cv::Rect> detect(const cv::Mat& image);

    private:
        std::vector<cv::Rect> haar_detection(const cv::Mat& image);
        std::vector<cv::Rect> caffe_detection(const cv::Mat& image);

        DetectionMethod detection_method;
        std::string model_path;
};

#endif