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
     * @param rho – The distance between the origin (top-left corner of an image) and the line
     * @param theta – The angle theta formed betwe 
     * 
     * 
     * 
     * 
     * 
     *
    */
    std::vector<cv::Point> polarLine2cartPoints(double rho, double theta, double factor = 10000)
    {
        double x0 = rho * cos(theta);                                          // the x-coordinate of the intersection pt between the line itself and the perpendicular line where rho lands.
        double y0 = rho * sin(theta);                                          // the y-coordinate of the intersection pt between the line itself and the perpendicular line where rho lands.
        cv::Point p1(x0 + factor * (-sin(theta)), y0 + factor * (cos(theta))); // from the intersection point (x0, y0), go along the line leftward
        cv::Point p2(x0 - factor * (-sin(theta)), y0 - factor * (cos(theta))); // from the intersection point (x0, y0), go along the line rightward
        return std::vector<cv::Point>{p1, p2};
    }

    /**
     * 
     * 
     * 
     * 
     * 
     * 
     * 
     * 
    */
    std::vector<cv::Point> reverse_roi(const cv::Point &roi_topleft, const std::vector<cv::Point> &points)
    {
        std::vector<cv::Point> pts = std::vector<cv::Point>(points);

        std::for_each(pts.begin(), pts.end(), [&](cv::Point &pt)
                      {
                          pt.x += roi_topleft.x;
                          pt.y += roi_topleft.y;
                      });
        return pts;
    }
}
