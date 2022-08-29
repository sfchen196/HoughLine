cmake_minimum_required(VERSION 3.10.2)
project(opencv_example VERSION 0.0.0)
set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED On)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

message(${OpenCV_LIBS})
message(${OpenCV_INCLUDE_DIRS})




add_library(hough_lines SHARED hough_lines.cpp)
target_link_libraries(hough_lines PUBLIC ${OpenCV_LIBS} Eigen3::Eigen OpenMP::OpenMP_CXX)
target_include_directories(hough_lines PUBLIC ${OpenCV_INCLUDE_DIRS} )


add_executable(main main.cpp)
target_link_libraries(main hough_lines yaml-cpp)

add_executable(example_cowling example_code/example_with_cowling_images.cpp)
target_link_libraries(example_cowling hough_lines yaml-cpp)
add_executable(example example_code/example_with_example_images.cpp)
target_link_libraries(example hough_lines yaml-cpp)
# target_link_libraries(main ${OpenCV_LIBS} Eigen3::Eigen OpenMP::OpenMP_CXX)
# target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS} )
