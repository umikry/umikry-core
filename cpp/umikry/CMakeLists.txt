add_definitions("-std=c++11")

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})

set(SOURCE_FILES
  umikry.cpp
)

add_executable(umikry-core-app ${SOURCE_FILES})
set_target_properties(umikry-core-app PROPERTIES OUTPUT_NAME umikry)
target_link_libraries(umikry-core-app ${OpenCV_LIBS} ${CURL_LIBRARIES} umikrycore)