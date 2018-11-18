 cmake_minimum_required(VERSION 2.8.12)
 project(umikry)

 add_definitions("-std=c++11")

 include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
 conan_basic_setup()

 add_executable(umikry umikry.cpp)
 add_library(UmikryFaceDetector UmikryFaceDetector.cpp)

 target_link_libraries(umikry ${CONAN_LIBS} UmikryFaceDetector)