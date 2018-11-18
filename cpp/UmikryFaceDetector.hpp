#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

enum class DetectionMethod { HAAR, CAFFE };

class UmikryFaceDetector {
    public:
        UmikryFaceDetector(DetectionMethod detection_method);
        ~UmikryFaceDetector() {};

        std::map<int, std::vector<int>> detect(cv::Mat image);

    private:
        std::map<int, std::vector<int>> haar_detection(cv::Mat image);
        std::map<int, std::vector<int>> caffe_detection(cv::Mat image);

        DetectionMethod m_detection_method;
};

#endif