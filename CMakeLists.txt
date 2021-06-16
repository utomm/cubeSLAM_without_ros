cmake_minimum_required(VERSION 3.19)
project(cubeSLAM_without_ros)

set(CMAKE_CXX_STANDARD 17)


include_directories(detect_3d_cuboid/include
        line_lbd/include
        object_slam/include)
#find_package(PCL REQUIRED)

#macro(print_all_variables)
#    message(STATUS "print_all_variables------------------------------------------{")
#    get_cmake_property(_variableNames VARIABLES)
#    foreach (_variableName ${_variableNames})
#        message(STATUS "${_variableName}=${${_variableName}}")
#    endforeach ()
#    message(STATUS "print_all_variables------------------------------------------}")
#endmacro()
#
#print_all_variables()


add_subdirectory(./line_lbd)
add_subdirectory(./detect_3d_cuboid)
add_subdirectory(./object_slam)


# add_executable(cubeSLAM_without_ros main.cpp)