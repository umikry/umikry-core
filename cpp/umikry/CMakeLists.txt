add_definitions("-std=c++11")

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})

set(SOURCE_FILES
  umikry.cpp
)

add_executable(umikry-core ${SOURCE_FILES})
target_link_libraries(umikry-core ${OpenCV_LIBS} ${CURL_LIBRARIES} umikrycore)