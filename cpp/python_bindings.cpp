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

cv::Mat umikry(py::array_t<uint8_t> image, std::string model_path, DetectionMethod detectionMethod, TransformationMethod transformationMethod) {    
    // TODO check dims

    py::buffer_info buffer = image.request();
    
    int rows = buffer.shape[0];
    int cols = buffer.shape[1];
    int channels = buffer.shape[2];

    if (channels != 3) {
        throw std::invalid_argument("The image needs to have 3 channels.");
    }

    unsigned char* data = (unsigned char*) buffer.ptr;
    cv::Mat cv_image(rows, cols, CV_8UC3, cv::Scalar::all(0));

    cv_image.data = data;
    
    UmikryFaceDetector umikryFaceDetector = UmikryFaceDetector(DetectionMethod::CAFFE, model_path);
    UmikryFaceTransformator umikryFaceTransformator = UmikryFaceTransformator(TransformationMethod::BLUR);
 
    std::vector<cv::Rect> faces = umikryFaceDetector.detect(cv_image);
    umikryFaceTransformator.transform(cv_image, faces);
    
    return cv_image;
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

    py::class_<cv::Mat>(m, "Image", py::buffer_protocol())
        .def_buffer([](cv::Mat& image) -> py::buffer_info {
            return py::buffer_info(
                image.data,
                sizeof(unsigned char),
                py::format_descriptor<unsigned char>::format(),
                3,
                { image.rows, image.cols, image.channels() },
                { sizeof(unsigned char) * image.channels() * image.cols,
                  sizeof(unsigned char) * image.channels(),
                  sizeof(unsigned char) }
            );
        });

  m.attr("__version__") = "0.1.alpha";
}