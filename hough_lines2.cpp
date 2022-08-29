#include "hough_lines.h"
namespace features
{
    template <typename Margin, typename Line>
    auto hough_line(Margin &margin, std::vector<Line> &OUT_lines, Line &OUT_best_line, double d_rho, double d_theta, int threshold, double min_theta, double max_theta) -> void
    {
        std::vector<cv::Point> margin_pts;
        if (std::is_same<Margin, cv::Mat>::value)
        {
            /* extract x,y coordinates of the "edge pixels" */
            for (int i = 0; i < margin.rows; i++)
            {
                for (int j = 0; j < margin.cols; j++)
                {
                    if (margin.at<uchar>(i, j) == 255)
                    {
                        margin_pts.push_back(cv::Point2i(j, i));
                    }
                }
            }
        }
        else if (std::is_same<Margin, std::vector<cv::Point>>::value)
        {
            (std::vector<cv::Point> &)margin_pts = margin;
        }
        else
        {
            // some error messages
        }

        /* given the step size [d_theta], NO. of different values of theta [n_theta] should be enough to cover the range between
        the minimum theta and the maximum theta */
        int n_theta = (int)ceil((max_theta - min_theta) / d_theta);

        /* given the step size [d_rho], No. of different values of rho [n_rho] should be enough to cover *DOUBLE* 
    the longest distance in the image, which is the diagonal length
        *DOUBLE*: to accomodate negative rho values */
        int max_x, max_y;
        if (std::is_same<Margin, cv::Mat>::value)
        {
            max_x = margin.cols;
            max_y = margin.rows;
        }
        else if (std::is_same<Margin, std::vector<cv::Point>>::value)
        {
            max_x = std::max_element(margin_pts.begin(), margin_pts.end(), [](cv::Point a, cv::Point b)
                                     { return a.x < b.x; })
                        ->x;
            max_y = std::max_element(margin_pts.begin(), margin_pts.end(), [](cv::Point a, cv::Point b)
                                     { return a.y < b.y; })
                        ->y;
        }
        int n_rho = (int)2 * ceil(sqrt(pow(max_x, 2) + pow(max_y, 2)) / d_rho);

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

                    if (std::is_same<Line, cv::Vec2d>::value)
                    {
                        OUT_lines.push_back(curr_line);
                        if (curr_votes > max_votes)
                        {
                            max_votes = curr_votes;
                            OUT_best_line = cv::Vec2d(curr_line);
                        }
                    }
                    else if (std::is_same<Line, std::vector<cv::Point>>::value)
                    {
                        std::vector<cv::Point> curr_line_pts = polarLine2cartPoints(curr_line[0], curr_line[1], std::max(max_x, max_y));
                        OUT_lines.push_back(curr_line_pts);
                        if (curr_votes > max_votes)
                        {
                            max_votes = curr_votes;
                            OUT_best_line = std::vector<cv::Point>(curr_line_pts);
                        }
                    }
                    else
                    {
                        // some error messages
                    }
                }
            }
        }
    }
    template void hough_line<cv::Mat, cv::Vec2d>(cv::Mat &margin, std::vector<cv::Vec2d> &OUT_lines, cv::Vec2d &OUT_best_line, double d_rho, double d_theta, int threshold, double min_theta, double max_theta);
    // template void hough_line


    //     auto hough_line(cv::Mat &image, std::vector<cv::Point2i> &edge_locations_vec, std::vector<cv::Vec2d> &lines, cv::Vec2d &max_line, double d_rho, double d_theta, int threshold, double min_theta, double max_theta) -> void
    //     {
    //         /* given the step size [d_theta], NO. of different values of theta [n_theta] should be enough to cover the range between
    //         the minimum theta and the maximum theta */
    //         int n_theta = (int)ceil((max_theta - min_theta) / d_theta);

    //         /* given the step size [d_rho], No. of different values of rho [n_rho] should be enough to cover *DOUBLE*
    //     the longest distance in the image, which is the diagonal length
    //         *DOUBLE*: to accomodate negative rho values */
    //         int n_rho = (int)2 * ceil(sqrt(pow(image.rows, 2) + pow(image.cols, 2)) / d_rho);

    //         /* 4. prepare the accumulator matrix, initialized as zeros */
    //         cv::Mat A = cv::Mat::zeros(n_rho, n_theta, CV_16UC1);

    //         /* 5. construct the vector of different angles ranging from 0 to pi radians */
    //         double angles[n_theta];
    //         for (int i = 0; i < n_theta; i++)
    //         {
    //             angles[i] = i * d_theta + min_theta;
    //         }

    //         /* 6. loop thru each edge pixel and increment the corresponding entries in the accumulator matrix
    //         by calculating the formula for each line passing thru the pixel at each of the angles */
    //         for (const auto &pt : edge_locations_vec)
    //         {
    // #pragma omp parallel for shared(A)
    //             for (int j = 0; j < n_theta; j++)
    //             {
    //                 double x = (double)pt.x;
    //                 double y = (double)pt.y;
    //                 double theta = angles[j];
    //                 double rho = x * cos(theta) + y * sin(theta);
    //                 // divides RHO by D_RHO to fit into the accumulator matrix of N_RHO,
    //                 // and then increments RHO by N_RHO/2 to avoid negative values
    //                 int rho_index = (int)round(rho / d_rho + n_rho / 2);

    //                 A.at<uint16_t>(rho_index, j)++;
    //             }
    //         }

    //         /* 7. filter lines with votes above threshold */
    //         uint16_t max_votes = 0;
    //         uint16_t curr_votes = 0;
    //         for (int i = 0; i < A.rows; i++)
    //         {
    //             for (int j = 0; j < A.cols; j++)
    //             {
    //                 curr_votes = A.at<uint16_t>(i, j);
    //                 if (curr_votes > (uint16_t)threshold)
    //                 {
    //                     // revert the rho_index to the actual value of rho by subtraction and then multiplication
    //                     // revert the theta_index to the actual value of theta by multiplication and then addition
    //                     cv::Vec2d curr_line((i - n_rho / 2) * d_rho, j * d_theta + min_theta);
    //                     lines.push_back(curr_line);
    //                     if (curr_votes > max_votes)
    //                     {
    //                         max_votes = curr_votes;
    //                         max_line = curr_line;
    //                     }
    //                 }
    //             }
    //         }
    //     }

    auto polarLine2cartPoints(double rho, double theta, double factor) -> std::vector<cv::Point>
    {
        double x0 = rho * cos(theta);                                          // the x-coordinate of the intersection pt between the line itself and the perpendicular line where rho lands.
        double y0 = rho * sin(theta);                                          // the y-coordinate of the intersection pt between the line itself and the perpendicular line where rho lands.
        cv::Point p1(x0 + factor * (-sin(theta)), y0 + factor * (cos(theta))); // from the intersection point (x0, y0), go along the line leftward
        cv::Point p2(x0 - factor * (-sin(theta)), y0 - factor * (cos(theta))); // from the intersection point (x0, y0), go along the line rightward
        return std::vector<cv::Point>{p1, p2};
    }

    auto reverse_roi(const cv::Point &roi_topleft, const std::vector<cv::Point> &points) -> std::vector<cv::Point>
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