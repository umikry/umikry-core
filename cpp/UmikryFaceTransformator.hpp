#ifndef UMIKRY_TRANSFORMATOR_HPP
#define UMIKRY_TRANSFORMATOR_HPP

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

enum class TransformationMethod { BLUR };

class UmikryFaceTransformator {
    public:
        UmikryFaceTransformator(TransformationMethod transformation_method);
        ~UmikryFaceTransformator() {};

        void transform(cv::Mat& image, const std::vector<cv::Rect>& faces);
    private:
        void blur_transformation(cv::Mat& image, const std::vector<cv::Rect>& faces);
        TransformationMethod transformation_method;
};

#endif