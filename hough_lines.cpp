#include "hough_lines.h"
namespace features
{
    auto polarLine2cartPoints(double rho, double theta, double factor) -> std::vector<cv::Point>
    {
        double x0 = rho * cos(theta);                                          // the x-coordinate of the intersection pt between the line itself and the perpendicular line where rho lands.
        double y0 = rho * sin(theta);                                          // the y-coordinate of the intersection pt between the line itself and the perpendicular line where rho lands.
        cv::Point p1(x0 + factor * (-sin(theta)), y0 + factor * (cos(theta))); // from the intersection point (x0, y0), go along the line leftward
        cv::Point p2(x0 - factor * (-sin(theta)), y0 - factor * (cos(theta))); // from the intersection point (x0, y0), go along the line rightward
        return std::vector<cv::Point>{p1, p2};
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