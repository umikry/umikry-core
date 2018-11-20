#include "UmikryFaceDetector.hpp"

UmikryFaceDetector::UmikryFaceDetector(DetectionMethod detection_method, std::string model_path) {
    this->detection_method = detection_method;
    this->model_path = model_path;
}

std::vector<cv::Rect> UmikryFaceDetector::detect(cv::Mat image) {
    if (detection_method == DetectionMethod::HAAR) {
        return haar_detection(image);
    } else if (detection_method == DetectionMethod::CAFFE) {
        return caffe_detection(image);
    } else {
        return {};
    }
}

std::vector<cv::Rect> UmikryFaceDetector::haar_detection(cv::Mat image) {
    std::cout << "The method haar_detect() is not implemented yet." << std::endl;
    return {};
}

std::vector<cv::Rect> UmikryFaceDetector::caffe_detection(cv::Mat image) {
    cv::Mat blob = cv::dnn::blobFromImage(image, 0.4, image.size(), cv::Scalar(104, 117, 123), false, false);
    const std::string caffeConfigFile = model_path + "/deploy.prototxt";
    const std::string caffeWeightFile = model_path + "/res10_300x300_ssd_iter_140000_fp16.caffemodel";

    cv::dnn::Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);

    net.setInput(blob, "data");
    cv::Mat detection = net.forward("detection_out");
     
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    
    std::vector<cv::Rect> bounding_boxes = {};
    for(int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);
     
        if(confidence > 0.4) {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);
     
            cv::Rect bounding_box (x1, y1, x2 - x1, y2 - y1);
            bounding_boxes.push_back(bounding_box);
        }
    }
    return bounding_boxes;
}