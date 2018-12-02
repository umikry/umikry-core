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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>
#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "UmikryFaceDetector.hpp"
#include "UmikryFaceTransformator.hpp"

namespace py = pybind11;

py::array_t<uint8_t> umikry(py::array_t<uint8_t> input, const std::string &model_path, DetectionMethod detectionMethod, TransformationMethod transformationMethod) {
	if (input.ndim() != 3) {
		throw std::invalid_argument("The image needs to have 3 dimensions.");
	}

    if (input.shape()[2] != 3) {
        throw std::invalid_argument("The image needs to have 3 channels.");
    }

	cv::Mat cv_image(static_cast<int>(input.shape()[0]), static_cast<int>(input.shape()[1]), CV_8UC3);
	memcpy(cv_image.data, input.data(), input.nbytes());

	UmikryFaceDetector umikryFaceDetector = UmikryFaceDetector(DetectionMethod::CAFFE, model_path);
	UmikryFaceTransformator umikryFaceTransformator = UmikryFaceTransformator(TransformationMethod::BLUR);
	umikryFaceTransformator.transform(cv_image, umikryFaceDetector.detect(cv_image));

	auto result = py::array_t<uint8_t>(std::vector<ssize_t>(input.shape(), input.shape() + input.ndim()));
	memcpy(result.mutable_data(), cv_image.data, input.nbytes());
	return result;
}

py::array_t<uint8_t> umikry(py::array_t<uint8_t> input, DetectionMethod detectionMethod, TransformationMethod transformationMethod) {
	return umikry(input, ".", detectionMethod, transformationMethod);
}

py::array_t<uint8_t> umikry(py::array_t<uint8_t> input) {
	return umikry(input, ".", DetectionMethod::CAFFE, TransformationMethod::BLUR);
}

PYBIND11_MODULE(umikry, m) {

    m.doc() = R"pbdoc(
    umikry module
    -----------------------
    .. currentmodule:: umikry
    .. autosummary::
    :toctree: _generate
    umikry
    )pbdoc";

	py::enum_<DetectionMethod>(m, "DetectionMethod")
        .value("HAAR", DetectionMethod::HAAR)
        .value("CAFFE", DetectionMethod::CAFFE)
        .export_values();

    py::enum_<TransformationMethod>(m, "TransformationMethod")
        .value("BLUR", TransformationMethod::BLUR)
        .export_values();

	m.def("umikry", py::overload_cast<py::array_t<uint8_t>, const std::string&, DetectionMethod, TransformationMethod>(&umikry), R"pbdoc(transforms faces found by the umikry face detector)pbdoc");
	m.def("umikry", py::overload_cast<py::array_t<uint8_t>, DetectionMethod, TransformationMethod>(&umikry), R"pbdoc(transforms faces found by the umikry face detector)pbdoc");
	m.def("umikry", py::overload_cast<py::array_t<uint8_t>>(&umikry), R"pbdoc(transforms faces found by the umikry face detector)pbdoc");

  m.attr("__version__") = "0.1.alpha";
}