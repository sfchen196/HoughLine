#include <chrono>
#include <iostream>
#include <omp.h>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>

namespace features
{
    /**
     * @brief 
     * @param image – 8-bit, single-channel binary source image. The image may be modified by the function.
     * @param lines – Output vector of lines. Each line is represented by a 2-element vector, theta and rho.
     * @param max_line – The line with the highest votes. If multiple lines have the same highest vote, only the first one encountered is chosen.
     * @param d_rho – Distance resolution of the accumulator in pixels. 
     * @param d_theta – Angle resolution of the accumulator in radians.
     * @param threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes
     * @param min_theta – Minimum angle to check for lines. Must fall between 0 and max_theta.
     * @param max_theta – Maximum angle to check for lines. Must fall between min_theta and CV_PI.
     * @return void
    */
    auto hough_line(cv::Mat &image,
                    std::vector<cv::Vec2d> &lines,
                    cv::Vec2d &max_line,
                    double d_rho,
                    double d_theta,
                    int threshold,
                    double min_theta = 0,
                    double max_theta = CV_PI) -> void;

    /**
     * @brief 
     * @param image – 8-bit, single-channel binary source image. The image may be modified by the function.
     * 
     * @param edge_locations_vec – A STL library vector of cv::Point objects, can be the output of cv::findContours()
     * @param lines – Output vector of lines. Each line is represented by a 2-element vector, theta and rho.
     * @param max_line – The line with the highest votes. If multiple lines have the same highest vote, only the first one encountered is chosen.
     * @param d_rho – Distance resolution of the accumulator in pixels. 
     * @param d_theta – Angle resolution of the accumulator in radians.
     * @param threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes
     * @param min_theta – Minimum angle to check for lines. Must fall between 0 and max_theta.
     * @param max_theta – Maximum angle to check for lines. Must fall between min_theta and CV_PI.
     * @return void
    */
    auto hough_line(cv::Mat &image,
                    std::vector<cv::Point2i> &edge_locations_vec,
                    std::vector<cv::Vec2d> &lines,
                    cv::Vec2d &max_line,
                    double d_rho,
                    double d_theta,
                    int threshold,
                    double min_theta = 0,
                    double max_theta = CV_PI) -> void;


    /**
     * @brief 
     * @param image – 8-bit, single-channel binary source image. The image may be modified by the function.
     * 
     * @param lines – Output vector of lines. Each line is represented by two cv::Point2i points.
     * @param max_line – The line with the highest votes. If multiple lines have the same highest vote, only the first one encountered is chosen.
     * @param d_rho – Distance resolution of the accumulator in pixels. 
     * @param d_theta – Angle resolution of the accumulator in radians.
     * @param threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes
     * @param min_theta – Minimum angle to check for lines. Must fall between 0 and max_theta.
     * @param max_theta – Maximum angle to check for lines. Must fall between min_theta and CV_PI.
     * @return void
    */
    auto hough_line(cv::Mat &image,
                    std::vector<std::vector<cv::Point>> &lines,
                    std::vector<cv::Point> &max_line,
                    double d_rho,
                    double d_theta,
                    int threshold,
                    double min_theta = 0,
                    double max_theta = CV_PI) -> void;
    
        /**
     * @brief 
     * @param image – 8-bit, single-channel binary source image. The image may be modified by the function.
     * 
     * @param edge_locations_vec – A STL library vector of cv::Point objects, can be the output of cv::findContours()
     * @param lines – Output vector of lines. Each line is represented by two cv::Point2i points.
     * @param max_line – The line with the highest votes. If multiple lines have the same highest vote, only the first one encountered is chosen.
     * @param d_rho – Distance resolution of the accumulator in pixels. 
     * @param d_theta – Angle resolution of the accumulator in radians.
     * @param threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes
     * @param min_theta – Minimum angle to check for lines. Must fall between 0 and max_theta.
     * @param max_theta – Maximum angle to check for lines. Must fall between min_theta and CV_PI.
     * @return void
    */
    auto hough_line(cv::Mat &image,
                    std::vector<cv::Point2i> &edge_locations_vec,
                    std::vector<std::vector<cv::Point>> &lines,
                    std::vector<cv::Point> &max_line,
                    double d_rho,
                    double d_theta,
                    int threshold,
                    double min_theta = 0,
                    double max_theta = CV_PI) -> void;
}
