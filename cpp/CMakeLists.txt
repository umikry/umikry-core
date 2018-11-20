cmake_minimum_required(VERSION 2.8.12)
project(umikry)

find_package( OpenCV REQUIRED COMPONENTS core highgui imgproc dnn )
add_definitions("-std=c++11")

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(umikry umikry.cpp)
add_library(UmikryFaceDetector UmikryFaceDetector.cpp)

target_link_libraries(umikry ${OpenCV_LIBS} UmikryFaceDetector)