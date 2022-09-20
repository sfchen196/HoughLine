cmake_minimum_required(VERSION 3.10.2)
set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED On)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)


# Locate GTest
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})


add_library(utils SHARED ../utils.cpp)
target_link_libraries(utils PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
target_include_directories(utils PUBLIC ${OpenCV_INCLUDE_DIRS})

add_library(hough_lines SHARED ../hough_lines.cpp)
target_link_libraries(hough_lines PRIVATE utils)

add_library(tools SHARED tools.cpp)
target_link_libraries(tools PUBLIC ${OpenCV_LIBS})
target_include_directories(tools PUBLIC ${OpenCV_INCLUDE_DIRS})

add_executable(detect_lines_in_cap ./detect_lines_in_cap.cpp)
target_link_libraries(detect_lines_in_cap PUBLIC tools ${OpenCV_LIBS} hough_lines yaml-cpp)

add_executable(detect_circles_and_lines ./detect_circles_and_lines.cpp)
target_link_libraries(detect_circles_and_lines PUBLIC tools ${OpenCV_LIBS} hough_lines yaml-cpp)
