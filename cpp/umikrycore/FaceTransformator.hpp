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

#ifndef UMIKRY_TRANSFORMATOR_HPP
#define UMIKRY_TRANSFORMATOR_HPP

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace umikry {

enum class TransformationMethod { BLUR };

class FaceTransformator {
public:
	FaceTransformator(TransformationMethod transformation_method);
	~FaceTransformator() {};

	void transform(cv::Mat& image, const std::vector<cv::Rect>& faces);

	inline TransformationMethod getTransformationMethod() const { return transformationMethod; }
private:
	void blurTransformation(cv::Mat& image, const std::vector<cv::Rect>& faces);
	
	const TransformationMethod transformationMethod;
};

}

#endif