#include "UmikryFaceDetector.hpp"

UmikryFaceDetector::UmikryFaceDetector(DetectionMethod detection_method) {
    m_detection_method = detection_method;
}

std::map<int, std::vector<int>> UmikryFaceDetector::detect(cv::Mat image) {
    if (m_detection_method == DetectionMethod::HAAR) {
        return haar_detection(image);
    } else if (m_detection_method == DetectionMethod::CAFFE) {
        return caffe_detection(image);
    } else {
        return {};
    }
}

std::map<int, std::vector<int>> UmikryFaceDetector::haar_detection(cv::Mat image) {
    std::cout << "The method haar_detect() is not implemented yet." << std::endl;
    return {};
}

std::map<int, std::vector<int>> UmikryFaceDetector::caffe_detection(cv::Mat image) {
    // TODO: Implement umikry caffe detection
    std::cout << "The method caffe_detect() is just a placeholer at the moment." << std::endl;
    std::map<int, std::vector<int>> bounding_boxes = {
        {0, {100,50,150,100}}, 
        {1, {20, 20, 40, 50}}
    };
    return bounding_boxes;
}