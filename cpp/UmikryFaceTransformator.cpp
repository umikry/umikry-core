#include "UmikryFaceTransformator.hpp"

UmikryFaceTransformator::UmikryFaceTransformator(TransformationMethod transformation_method) {
    this->transformation_method = transformation_method;
}

void UmikryFaceTransformator::transform(cv::Mat& image, const std::vector<cv::Rect> faces) {
    if (transformation_method == TransformationMethod::BLUR) {
        blur_transformation(image, faces);
    }
}

void UmikryFaceTransformator::blur_transformation(cv::Mat& image, const std::vector<cv::Rect> faces) {
	for (cv::Rect face : faces) {
		if ((face.y + face.height) < image.rows && (face.x + face.width) < image.cols) {
			cv::blur(image(face), image(face), cv::Size(25, 25));
		}
	}
}