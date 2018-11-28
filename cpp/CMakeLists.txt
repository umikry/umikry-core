cmake_minimum_required(VERSION 2.8.12)
project(umikry)

find_package(pybind11 REQUIRED)
find_package( OpenCV REQUIRED COMPONENTS core highgui imgproc dnn )

add_definitions("-std=c++11")

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(UmikryCore UmikryCore.cpp)
pybind11_add_module(umikry python_bindings.cpp)

add_library(UmikryFaceDetector UmikryFaceDetector.cpp)
add_library(UmikryFaceTransformator UmikryFaceTransformator.cpp)

target_link_libraries(umikry PRIVATE ${OpenCV_LIBS} UmikryFaceDetector UmikryFaceTransformator)
target_link_libraries(UmikryCore ${OpenCV_LIBS} UmikryFaceDetector UmikryFaceTransformator)