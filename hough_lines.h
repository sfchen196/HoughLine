#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


namespace features
{

    /**
     * @brief
     * @param margin        – Input image or vector of points, containing information about where the image gradient changes abruptly.
     *                          Can be an 8-bit, single-channel binary source image (e.g. cv::Canny() output), or a vector of points (e.g. cv::findContours() output)
     * @param lines         – Output vector of lines. Each line is represented by a 2-element vector, theta and rho.
     * @param max_line      – The line with the highest votes. If multiple lines have the same highest vote, only the first one encountered is chosen.
     * @param d_rho         – Distance resolution of the accumulator in pixels. Must be positive.
     * @param d_theta       – Angle resolution of the accumulator in radians. Must be positive.
     * @param threshold     – Accumulator threshold parameter. Only those lines are returned that get enough votes
     * @param min_theta     – Minimum angle to check for lines. Both positive and negative values are allowed.
     * @param max_theta     – Maximum angle to check for lines. Both positive and negative values are allowed. MAX_THETA >= MIN_THETA 
     * @return void
     **/

    template <typename Margin>
    auto HoughLines(Margin &margin, std::vector<cv::Vec2d> &OUT_lines, cv::Vec2d &OUT_best_line, double d_rho, double d_theta, int threshold, double min_theta, double max_theta) -> void;

    /**
     * @brief
     * @param rho       – The distance between the origin (top-left corner of an image) and the line
     * @param theta     – The angle theta formed betwe 
     * @param factor    – The larger the FACTOR, the further away the calculated points are
     * @return          – A pair of points on the line
     **/
    auto polarLine2cartPoints(double rho, double theta, double factor = 10000) -> std::vector<cv::Point>;

    /**
     * @brief
     * @param roi_topleft   – The point at the top left corner of the roi
     * @param points        – A vector of points with respect to the roi
     * @return              – The same vector of points with respect to the whole image
     **/
    auto reverseROI(const cv::Point &roi_topleft, const std::vector<cv::Point> &points) -> std::vector<cv::Point>;
}
