cmake_minimum_required(VERSION 2.8.12)
project(umikry)

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc dnn imgcodecs)

add_subdirectory(umikrycore)
add_subdirectory(umikry)
add_subdirectory(pyumikry)