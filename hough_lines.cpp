#include "hough_lines.h"

namespace utils
{
    template <typename Margin>
    auto input(Margin &margin, std::vector<cv::Point> &OUT_margin_pts, int &OUT_max_x, int &OUT_max_y) -> void;

    template <>
    auto input<cv::Mat>(cv::Mat &margin, std::vector<cv::Point> &OUT_margin_pts, int &OUT_max_x, int &OUT_max_y) -> void;

    template <>
    auto input<std::vector<cv::Point>>(std::vector<cv::Point> &margin, std::vector<cv::Point> &OUT_margin_pts, int &OUT_max_x, int &OUT_max_y) -> void;

    auto prepareAccumulatorMatrix(int max_x, int max_y, double d_rho, double min_theta, double max_theta, double d_theta) -> std::pair<int, int>;
    auto accumulate(std::vector<cv::Point> &margin_pts, double *angles, int n_theta, cv::Mat &A, double d_rho, int n_rho) -> void;
    auto extractLines(cv::Mat &A, std::vector<cv::Vec2d> &OUT_lines, cv::Vec2d &OUT_best_line, int threshold, double d_rho, int n_rho, double d_theta, double min_theta) -> void;
}
namespace features
{

    template <typename Margin>
    auto HoughLines(Margin &margin, std::vector<cv::Vec2d> &OUT_lines, cv::Vec2d &OUT_best_line, double d_rho, double d_theta, int threshold, double min_theta, double max_theta) -> void
    {
        std::vector<cv::Point> margin_pts;
        int max_x, max_y;
        utils::input(margin, margin_pts, max_x, max_y);

        const auto [n_rho, n_theta] = utils::prepareAccumulatorMatrix(max_x, max_y, d_rho, min_theta, max_theta, d_theta);

        /* 4. prepare the accumulator matrix, initialized as zeros */
        cv::Mat A = cv::Mat::zeros(n_rho, n_theta, CV_16UC1);

        /* 5. construct the array of different angles ranging from min_theta to max_theta */
        double angles[n_theta];
        for (int i = 0; i < n_theta; i++)
            angles[i] = i * d_theta + min_theta;

        utils::accumulate(margin_pts, angles, n_theta, A, d_rho, n_rho);

        utils::extractLines(A, OUT_lines, OUT_best_line, threshold, d_rho, n_rho, d_theta, min_theta);
    }
    template void HoughLines<cv::Mat>(cv::Mat &margin, std::vector<cv::Vec2d> &OUT_lines, cv::Vec2d &OUT_best_line, double d_rho, double d_theta, int threshold, double min_theta, double max_theta);
    template void HoughLines<std::vector<cv::Point>>(std::vector<cv::Point> &margin, std::vector<cv::Vec2d> &OUT_lines, cv::Vec2d &OUT_best_line, double d_rho, double d_theta, int threshold, double min_theta, double max_theta);

    auto polarLine2cartPoints(double rho, double theta, double factor) -> std::vector<cv::Point>
    {
        double x0 = rho * cos(theta);                                          // the x-coordinate of the intersection pt between the line itself and the perpendicular line where rho lands.
        double y0 = rho * sin(theta);                                          // the y-coordinate of the intersection pt between the line itself and the perpendicular line where rho lands.
        cv::Point p1(x0 + factor * (-sin(theta)), y0 + factor * (cos(theta))); // from the intersection point (x0, y0), go along the line leftward
        cv::Point p2(x0 - factor * (-sin(theta)), y0 - factor * (cos(theta))); // from the intersection point (x0, y0), go along the line rightward
        return std::vector<cv::Point>{p1, p2};
    }


    auto cartPoints2polarLine(cv::Point pt1, cv::Point pt2) -> cv::Vec2d
    {
        double theta = -1 * atan((pt1.x - pt2.x) / (pt1.y - pt2.y));
        double rho = pt1.y * sin(theta) + pt1.x * cos(theta);
        return cv::Vec2d{rho, theta};
    }

    auto reverseROI(const cv::Point &roi_topleft, const std::vector<cv::Point> &points) -> std::vector<cv::Point>
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