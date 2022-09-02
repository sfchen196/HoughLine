#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace utils
{
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
        OUT_max_x = margin.cols - 1;
        OUT_max_y = margin.rows - 1;
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
    auto prepareAccumulatorMatrix(int max_x, int max_y, double d_rho, double min_theta, double max_theta, double d_theta) -> std::pair<int, int>
    {
        assert(d_rho > 0);
        assert(d_theta > 0);
        assert(max_theta >= min_theta);

        /* given the step size [d_rho], No. of different values of rho [n_rho] should be enough to cover *DOUBLE* 
            the longest distance in the image, which is the diagonal length
                    *DOUBLE*: to accomodate negative rho values */
        int n_rho = (int)(2 * ceil(sqrt(pow(max_x, 2) + pow(max_y, 2)) / d_rho) + 1);

        /* given the step size [d_theta], NO. of different values of theta [n_theta] should be enough to cover the range between
             the minimum theta and the maximum theta */
        int n_theta = (int)(ceil((max_theta - min_theta) / d_theta) + 1);
        return std::pair<int, int>{n_rho, n_theta};
    }

    auto accumulate(std::vector<cv::Point> &margin_pts, double *angles, int n_theta, cv::Mat &A, double d_rho, int n_rho) -> void
    {
        ////////////// input: margin_pts, empty accumulator matrix A, angles
        /* 6. loop thru each edge pixel and increment the corresponding entries in the accumulator matrix
        by calculating the formula for each line passing thru the pixel at each of the angles */
        for (const auto &pt : margin_pts)
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
    }

    auto extractLines(cv::Mat &A, std::vector<cv::Vec2d> &OUT_lines, cv::Vec2d &OUT_best_line, int threshold, double d_rho, int n_rho, double d_theta, double min_theta) -> void
    {
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
}