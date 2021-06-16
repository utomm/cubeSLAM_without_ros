cmake_minimum_required(VERSION 3.19)
project(cubeSLAM_without_ros)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(./line_lbd)
add_subdirectory(./detect_3d_cuboid)


# add_executable(cubeSLAM_without_ros main.cpp)