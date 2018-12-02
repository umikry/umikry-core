if (UNIX)
  find_package( CURL REQUIRED ) 
endif (UNIX)

add_definitions("-std=c++11")

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES
  UmikryFaceDetector.cpp
  UmikryFaceDetector.hpp
  UmikryFaceTransformator.cpp
  UmikryFaceTransformator.hpp
)

add_library(umikrycore ${SOURCE_FILES})
target_link_libraries(umikrycore)