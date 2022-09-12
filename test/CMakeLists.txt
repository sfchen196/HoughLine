cmake_minimum_required(VERSION 3.10.2)
set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED On)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)


# Locate GTest
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(utils SHARED ../utils.cpp)
target_link_libraries(utils PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
target_include_directories(utils PUBLIC ${OpenCV_INCLUDE_DIRS} )

add_library(hough_lines SHARED ../hough_lines.cpp)
target_link_libraries(hough_lines PRIVATE utils)


add_executable(InputTests InputTests.cpp)
target_link_libraries(InputTests PUBLIC ${OpenCV_LIBS} hough_lines yaml-cpp ${GTEST_LIBRARIES} pthread)

add_executable(PrepareAMTests PrepareAMTests.cpp)
target_link_libraries(PrepareAMTests PUBLIC ${OpenCV_LIBS} hough_lines yaml-cpp ${GTEST_LIBRARIES} pthread)

add_executable(AccumulateTests AccumulateTests.cpp)
target_link_libraries(AccumulateTests PUBLIC ${OpenCV_LIBS} hough_lines yaml-cpp ${GTEST_LIBRARIES} pthread)

add_executable(ExtractLinesTests ExtractLinesTests.cpp)
target_link_libraries(ExtractLinesTests PUBLIC ${OpenCV_LIBS} hough_lines yaml-cpp ${GTEST_LIBRARIES} pthread)