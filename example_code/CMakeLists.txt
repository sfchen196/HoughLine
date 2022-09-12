cmake_minimum_required(VERSION 3.10.2)
project(opencv_example VERSION 0.0.0)
set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED On)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)


find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})


add_library(utils SHARED ../utils.cpp)
target_link_libraries(utils PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
target_include_directories(utils PUBLIC ${OpenCV_INCLUDE_DIRS} )

add_library(hough_lines SHARED ../hough_lines.cpp)
target_link_libraries(hough_lines PRIVATE utils)


add_executable(example_cowling example_with_cowling_images.cpp)
target_link_libraries(example_cowling PUBLIC ${OpenCV_LIBS} hough_lines yaml-cpp)

add_executable(example_sample example_with_sample_images.cpp)
target_link_libraries(example_sample PUBLIC ${OpenCV_LIBS} hough_lines yaml-cpp)

