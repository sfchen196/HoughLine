#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <vector>

template <typename Margin>
auto input(Margin &margin, std::vector<cv::Point> &OUT_margin_pts, int &OUT_max_x, int &OUT_max_y) -> void;

template <>
auto input<cv::Mat>(cv::Mat &margin, std::vector<cv::Point> &OUT_margin_pts, int &OUT_max_x, int &OUT_max_y) -> void;

template <>
auto input<std::vector<cv::Point>>(std::vector<cv::Point> &margin, std::vector<cv::Point> &OUT_margin_pts, int &OUT_max_x, int &OUT_max_y) -> void;

auto prepareAccumulatorMatrix(int max_x, int max_y, double d_rho, double min_theta, double max_theta, double d_theta) -> std::pair<int, int>;
auto accumulate(std::vector<cv::Point> &margin_pts, double *angles, int n_theta, cv::Mat &A, double d_rho, int n_rho) -> void;
auto extractLines(cv::Mat &A, std::vector<cv::Vec2d> &OUT_lines, cv::Vec2d &OUT_best_line, int threshold, double d_rho, int n_rho, double d_theta, double min_theta) -> void;
