#include <pybind11/pybind11.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "UmikryFaceDetector.hpp"
#include "UmikryFaceTransformator.hpp"

cv::Mat umikry(cv::Mat image, std::string model_path, DetectionMethod detectionMethod, TransformationMethod transformationMethod) {    
    UmikryFaceDetector umikryFaceDetector = UmikryFaceDetector(DetectionMethod::CAFFE, model_path);
    UmikryFaceTransformator umikryFaceTransformator = UmikryFaceTransformator(TransformationMethod::BLUR);
    
    std::vector<cv::Rect> faces = umikryFaceDetector.detect(image);
    umikryFaceTransformator.transform(image, faces);
    
    return image;
}

namespace py = pybind11;

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

  m.attr("__version__") = "dev";
}