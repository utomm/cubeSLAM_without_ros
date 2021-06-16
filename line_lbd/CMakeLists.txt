cmake_minimum_required(VERSION 2.8.3)
project(line_lbd)


set(CMAKE_PREFIX_PATH "/home/hu/install/lib/cmake/opencv4")
find_package(OpenCV 4.0 REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)


## Declare a C++ library
add_library(line_lbd_lib
  libs/binary_descriptor.cpp libs/binary_descriptor_matcher.cpp libs/LSDDetector.cpp  libs/lsd.cpp libs/bitarray.hpp  
  libs/bitops.hpp libs/draw.cpp libs/precomp.hpp libs/types.hpp  class/line_lbd_allclass.cpp
)

add_executable(linelbd_detect_node src/detect_lines.cpp)
target_link_libraries(linelbd_detect_node  line_lbd_lib ${OpenCV_LIBRARIES})
