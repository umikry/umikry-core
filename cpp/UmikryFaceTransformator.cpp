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

#include "UmikryFaceTransformator.hpp"

UmikryFaceTransformator::UmikryFaceTransformator(TransformationMethod transformation_method) {
    this->transformation_method = transformation_method;
}

void UmikryFaceTransformator::transform(cv::Mat& image, const std::vector<cv::Rect>& faces) {
    if (transformation_method == TransformationMethod::BLUR) {
        blur_transformation(image, faces);
    }
}

void UmikryFaceTransformator::blur_transformation(cv::Mat& image, const std::vector<cv::Rect>& faces) {
	for (cv::Rect face : faces) {
		if ((face.y + face.height) < image.rows && (face.x + face.width) < image.cols) {
			cv::blur(image(face), image(face), cv::Size(25, 25));
		}
	}
}