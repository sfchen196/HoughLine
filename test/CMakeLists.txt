cmake_minimum_required(VERSION 3.10.2)
set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED On)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)


# Locate GTest
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})


add_library(hough_lines SHARED ../hough_lines.cpp)
target_link_libraries(hough_lines PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
target_include_directories(hough_lines PUBLIC ${OpenCV_INCLUDE_DIRS} )


add_executable(InputTests InputTests.cpp)
target_link_libraries(InputTests PUBLIC hough_lines yaml-cpp ${GTEST_LIBRARIES} pthread)
target_include_directories(InputTests PUBLIC ${GTEST_INCLUDE_DIRS})
add_executable(PrepareAMTests PrepareAMTests.cpp)
target_link_libraries(PrepareAMTests PUBLIC hough_lines yaml-cpp ${GTEST_LIBRARIES} pthread)
target_include_directories(PrepareAMTests PUBLIC ${GTEST_INCLUDE_DIRS})
add_executable(AccumulateTests AccumulateTests.cpp)
target_link_libraries(AccumulateTests PUBLIC hough_lines yaml-cpp ${GTEST_LIBRARIES} pthread)
target_include_directories(AccumulateTests PUBLIC ${GTEST_INCLUDE_DIRS})
add_executable(ExtractLinesTests ExtractLinesTests.cpp)
target_link_libraries(ExtractLinesTests PUBLIC hough_lines yaml-cpp ${GTEST_LIBRARIES} pthread)
target_include_directories(ExtractLinesTests PUBLIC ${GTEST_INCLUDE_DIRS})