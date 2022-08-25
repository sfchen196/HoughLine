#include <chrono>
#include <iostream>
#include <omp.h>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>

namespace features
{
    template <typename Margin>
    auto HoughLines_input(Margin &margin, std::vector<cv::Point> &OUT_margin_pts, int &OUT_max_x, int &OUT_max_y) -> void
    {
        // some error messages
    }
    template <>
    auto HoughLines_input<cv::Mat>(cv::Mat &margin, std::vector<cv::Point> &OUT_margin_pts, int &OUT_max_x, int &OUT_max_y) -> void
    {
        /* extract x,y coordinates of the "edge pixels" */
        for (int i = 0; i < margin.rows; i++)
        {
            for (int j = 0; j < margin.cols; j++)
            {
                if (margin.at<uchar>(i, j) == 255)
                {
                    OUT_margin_pts.push_back(cv::Point2i(j, i));
                }
            }
        }
        /* given the step size [d_rho], No. of different values of rho [n_rho] should be enough to cover *DOUBLE* 
            the longest distance in the image, which is the diagonal length
                        *DOUBLE*: to accomodate negative rho values */
        OUT_max_x = margin.cols;
        OUT_max_y = margin.rows;
    }
    template <>
    auto HoughLines_input<std::vector<cv::Point>>(std::vector<cv::Point> &margin, std::vector<cv::Point> &OUT_margin_pts, int &OUT_max_x, int &OUT_max_y) -> void
    {
        OUT_margin_pts = margin;
        /* given the step size [d_rho], No. of different values of rho [n_rho] should be enough to cover *DOUBLE* 
            the longest distance in the image, which is the diagonal length
                    *DOUBLE*: to accomodate negative rho values */
        OUT_max_x = std::max_element(OUT_margin_pts.begin(), OUT_margin_pts.end(), [](cv::Point a, cv::Point b)
                                     { return a.x < b.x; })
                        ->x;
        OUT_max_y = std::max_element(OUT_margin_pts.begin(), OUT_margin_pts.end(), [](cv::Point a, cv::Point b)
                                     { return a.y < b.y; })
                        ->y;
    }

    template <typename Margin>
    auto HoughLines(Margin &margin, std::vector<cv::Vec2d> &OUT_lines, cv::Vec2d &OUT_best_line, double d_rho, double d_theta, int threshold, double min_theta, double max_theta) -> void
    {
        std::vector<cv::Point> OUT_margin_pts;
        int OUT_max_x, OUT_max_y;
        HoughLines_input(margin, OUT_margin_pts, OUT_max_x, OUT_max_y);
        int n_rho = (int)2 * ceil(sqrt(pow(OUT_max_x, 2) + pow(OUT_max_y, 2)) / d_rho);

        /* given the step size [d_theta], NO. of different values of theta [n_theta] should be enough to cover the range between
        the minimum theta and the maximum theta */
        int n_theta = (int)ceil((max_theta - min_theta) / d_theta);

        /* 4. prepare the accumulator matrix, initialized as zeros */
        cv::Mat A = cv::Mat::zeros(n_rho, n_theta, CV_16UC1);

        /* 5. construct the vector of different angles ranging from 0 to pi radians */
        double angles[n_theta];
        for (int i = 0; i < n_theta; i++)
        {
            angles[i] = i * d_theta + min_theta;
        }

        /* 6. loop thru each edge pixel and increment the corresponding entries in the accumulator matrix
        by calculating the formula for each line passing thru the pixel at each of the angles */
        for (const auto &pt : OUT_margin_pts)
        {
#pragma omp parallel for shared(A)
            for (int j = 0; j < n_theta; j++)
            {
                double x = (double)pt.x;
                double y = (double)pt.y;
                double theta = angles[j];
                double rho = x * cos(theta) + y * sin(theta);
                // divides RHO by D_RHO to fit into the accumulator matrix of N_RHO,
                // and then increments RHO by N_RHO/2 to avoid negative values
                int rho_index = (int)round(rho / d_rho + n_rho / 2);

                A.at<uint16_t>(rho_index, j)++;
            }
        }

        /* 7. filter lines with votes above threshold */
        uint16_t max_votes = 0;
        uint16_t curr_votes = 0;
        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < A.cols; j++)
            {
                curr_votes = A.at<uint16_t>(i, j);
                if (curr_votes > (uint16_t)threshold)
                {
                    // revert the rho_index to the actual value of rho by subtraction and then multiplication
                    // revert the theta_index to the actual value of theta by multiplication and then addition
                    cv::Vec2d curr_line((i - n_rho / 2) * d_rho, j * d_theta + min_theta);
                    OUT_lines.push_back(curr_line);
                    if (curr_votes > max_votes)
                    {
                        max_votes = curr_votes;
                        OUT_best_line = curr_line;
                    }
                }
            }
        }
    }

    // /**
    //  * @brief
    //  * @param image – 8-bit, single-channel binary source image. The image may be modified by the function.
    //  * @param lines – Output vector of lines. Each line is represented by a 2-element vector, theta and rho.
    //  * @param max_line – The line with the highest votes. If multiple lines have the same highest vote, only the first one encountered is chosen.
    //  * @param d_rho – Distance resolution of the accumulator in pixels.
    //  * @param d_theta – Angle resolution of the accumulator in radians.
    //  * @param threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes
    //  * @param min_theta – Minimum angle to check for lines. Must falls between 0 and max_theta.
    //  * @param max_theta – Maximum angle to check for lines. Must fall between min_theta and CV_PI.
    //  * @return void
    // */
    // auto hough_line(cv::Mat &image,
    //                 std::vector<cv::Vec2d> &lines,
    //                 cv::Vec2d &max_line,
    //                 double d_rho = 1,
    //                 double d_theta = CV_PI / 180,
    //                 int threshold = 100,
    //                 double min_theta = 0,
    //                 double max_theta = CV_PI) -> void;

    // /**
    //  * @brief
    //  * @param image – 8-bit, single-channel binary source image. The image may be modified by the function.
    //  *
    //  * @param edge_locations_vec – A STL library vector of cv::Point objects, can be the output of cv::findContours()
    //  * @param lines – Output vector of lines. Each line is represented by a 2-element vector, theta and rho.
    //  * @param max_line – The line with the highest votes. If multiple lines have the same highest vote, only the first one encountered is chosen.
    //  * @param d_rho – Distance resolution of the accumulator in pixels.
    //  * @param d_theta – Angle resolution of the accumulator in radians.
    //  * @param threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes
    //  * @param min_theta – Minimum angle to check for lines. Must fall between 0 and max_theta.
    //  * @param max_theta – Maximum angle to check for lines. Must fall between min_theta and CV_PI.
    //  * @return void
    // */
    // auto hough_line(cv::Mat &image,
    //                 std::vector<cv::Point2i> &edge_locations_vec,
    //                 std::vector<cv::Vec2d> &lines,
    //                 cv::Vec2d &max_line,
    //                 double d_rho,
    //                 double d_theta,
    //                 int threshold,
    //                 double min_theta = 0,
    //                 double max_theta = CV_PI) -> void;

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

    auto polarLine2cartPoints(double rho, double theta, double factor = 10000) -> std::vector<cv::Point>;

    auto reverse_roi(const cv::Point &roi_topleft, const std::vector<cv::Point> &points) -> std::vector<cv::Point>;

}
