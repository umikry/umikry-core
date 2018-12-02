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

PYBIND11_MODULE(umikry, m) {

    m.doc() = R"pbdoc(
    umikry module
    -----------------------
    .. currentmodule:: umikry
    .. autosummary::
    :toctree: _generate
    umikry
    )pbdoc";

    m.def("umikry", &umikry, R"pbdoc(
        transforms faces found by the umikry face detector
        )pbdoc");

    py::enum_<DetectionMethod>(m, "DetectionMethod")
        .value("HAAR", DetectionMethod::HAAR)
        .value("CAFFE", DetectionMethod::CAFFE)
        .export_values();

    py::enum_<TransformationMethod>(m, "TransformationMethod")
        .value("BLUR", TransformationMethod::BLUR)
        .export_values();

  m.attr("__version__") = "0.1.alpha";
}