if (UNIX)
find_package( CURL REQUIRED ) 
endif (UNIX)

add_definitions("-std=c++11")

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES
  FaceDetector.cpp
  FaceDetector.hpp
  FaceTransformator.cpp
  FaceTransformator.hpp
)

add_library(umikrycore ${SOURCE_FILES})
target_link_libraries(umikrycore ${OpenCV_LIBS} ${CURL_LIBRARIES})