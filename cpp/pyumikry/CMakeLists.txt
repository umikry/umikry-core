add_definitions("-std=c++11")

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})

set(SOURCE_FILES
  python_bindings.cpp
)

pybind11_add_module(umikry ${SOURCE_FILES})
target_link_libraries(umikry PRIVATE ${OpenCV_LIBS} ${CURL_LIBRARIES} umikrycore)