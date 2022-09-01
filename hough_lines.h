#include <chrono>
#include <iostream>
#include <omp.h>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>

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
    auto input(Margin &margin, std::vector<cv::Point> &OUT_margin_pts, int &OUT_max_x, int &OUT_max_y) -> void
    {
        throw std::invalid_argument("Invalid argument: Input argument MARGIN should be an instance of cv::Mat or std::vector<cv::Point>");
    }
    template <>
    auto input<cv::Mat>(cv::Mat &margin, std::vector<cv::Point> &OUT_margin_pts, int &OUT_max_x, int &OUT_max_y) -> void
    {
        /* extract x,y coordinates of the "edge pixels" */
        for (int i = 0; i < margin.rows; i++)
        {
            for (int j = 0; j < margin.cols; j++)
            {
                if (margin.at<uchar>(i, j) > 0)
                {
                    OUT_margin_pts.push_back(cv::Point2i(j, i));
                }
            }
        }
        OUT_max_x = margin.cols-1;
        OUT_max_y = margin.rows-1;
    }
    template <>
    auto input<std::vector<cv::Point>>(std::vector<cv::Point> &margin, std::vector<cv::Point> &OUT_margin_pts, int &OUT_max_x, int &OUT_max_y) -> void
    {
        OUT_margin_pts = margin;

        OUT_max_x = std::max_element(OUT_margin_pts.begin(), OUT_margin_pts.end(), [](cv::Point a, cv::Point b)
                                     { return a.x < b.x; })
                        ->x;
        OUT_max_y = std::max_element(OUT_margin_pts.begin(), OUT_margin_pts.end(), [](cv::Point a, cv::Point b)
                                     { return a.y < b.y; })
                        ->y;
    }

    auto prepareAccumulatorMatrix(int max_x, int max_y, double d_rho, double min_theta, double max_theta, double d_theta) -> std::pair<int, int>;

    auto accumulate(std::vector<cv::Point> &margin_pts, double *angles, int n_theta, cv::Mat &A, double d_rho, int n_rho) -> void; 

    auto extractLines(cv::Mat &A, std::vector<cv::Vec2d> &OUT_lines, cv::Vec2d &OUT_best_line, int threshold, double d_rho, int n_rho, double d_theta, double min_theta) -> void;

    /**
     * @brief
     * @param rho       – The distance between the origin (top-left corner of an image) and the line
     * @param theta     – The angle theta formed betwe 
     * @param factor    – The larger the FACTOR, the further away the calculated points are
     * @return          – A pair of points on the line
     **/

    template <typename Margin>
    auto HoughLines(Margin &margin, std::vector<cv::Vec2d> &OUT_lines, cv::Vec2d &OUT_best_line, double d_rho, double d_theta, int threshold, double min_theta, double max_theta) -> void
    {
        std::vector<cv::Point> margin_pts;
        int max_x, max_y;
        input(margin, margin_pts, max_x, max_y);

        const auto [n_rho, n_theta] = prepareAccumulatorMatrix(max_x, max_y, d_rho, min_theta, max_theta, d_theta);

        /* 4. prepare the accumulator matrix, initialized as zeros */
        cv::Mat A = cv::Mat::zeros(n_rho, n_theta, CV_16UC1);

        /* 5. construct the array of different angles ranging from min_theta to max_theta */
        double angles[n_theta];
        for (int i = 0; i < n_theta; i++)
            angles[i] = i * d_theta + min_theta;

        accumulate(margin_pts, angles, n_theta, A, d_rho, n_rho);

        extractLines(A, OUT_lines, OUT_best_line, threshold, d_rho, n_rho, d_theta, min_theta);
    }

    auto polarLine2cartPoints(double rho, double theta, double factor = 10000) -> std::vector<cv::Point>;

    /**
     * @brief
     * @param roi_topleft   – The point at the top left corner of the roi
     * @param points        – A vector of points with respect to the roi
     * @return              – The same vector of points with respect to the whole image
     **/
    auto reverseROI(const cv::Point &roi_topleft, const std::vector<cv::Point> &points) -> std::vector<cv::Point>;
}




