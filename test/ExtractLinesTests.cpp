#include "../hough_lines.h"
#include <gtest/gtest.h>

class AccumulateTest : public ::testing::Test
{
protected:
    cv::Mat img;
    std::vector<cv::Point> margin;

    void SetUp() override
    {
        img = cv::Mat::zeros(cv::Size(5, 5), CV_8UC1);
        img.forEach<uchar>([](uchar &pixel, const int *position) -> void
                           {
                               if (position[1] == 2)
                                   pixel = 255;
                           });
        margin.push_back(cv::Point(2, 0));
        margin.push_back(cv::Point(2, 1));
        margin.push_back(cv::Point(2, 2));
        margin.push_back(cv::Point(2, 3));
        margin.push_back(cv::Point(2, 4));
    }

    void TearDown() override
    {
    }
};

TEST_F(AccumulateTest, single_theta_value)
{
    int n_theta = 1, n_rho = 11, threshold = 0;
    double d_theta = CV_PI / 180, min_theta = 0, d_rho = 1;
    cv::Mat A = cv::Mat::zeros(n_rho, n_theta, CV_16UC1);
    A.at<uint16_t>(7, 0) = 5;

    std::vector<cv::Vec2d> OUT_lines, actual_lines;
    cv::Vec2d OUT_best_line, actual_best_line;
    actual_lines.push_back(cv::Vec2d(2, 0));
    actual_best_line = cv::Vec2d(2, 0);

    features::extractLines(A, OUT_lines, OUT_best_line, threshold, d_rho, n_rho, d_theta, min_theta);

    std::sort(OUT_lines.begin(), OUT_lines.end(), [](const cv::Vec2d &lhs, const cv::Vec2d &rhs)
              {
                  if (lhs[1] != rhs[1])
                      return lhs[1] < rhs[1];
                  return lhs[0] < rhs[0];
              });
    std::sort(actual_lines.begin(), actual_lines.end(), [](const cv::Vec2d &lhs, const cv::Vec2d &rhs)
              {
                  if (lhs[1] != rhs[1])
                      return lhs[1] < rhs[1];
                  return lhs[0] < rhs[0];
              });
    ASSERT_EQ(OUT_lines.size(), actual_lines.size()) << "Vectors OUT_lines and actual_lines have different sizes";
    for (size_t i = 0, size = OUT_lines.size(); i < size; i++)
    {
        EXPECT_EQ(OUT_lines[i], actual_lines[i]) << "Vectors OUT_lines and actual_lines are different";
    }
    EXPECT_EQ(OUT_best_line, actual_best_line) << "Vectors OUT_best_line and actual_best_line are different";
}

TEST_F(AccumulateTest, handles_negative_rho_values)
{
    int n_theta = 2, n_rho = 11, threshold = 0;
    double d_theta = CV_PI / 2, min_theta = -CV_PI / 2, d_rho = 1;
    cv::Mat A = cv::Mat::zeros(n_rho, n_theta, CV_16UC1);
    A.at<uint16_t>(7, 1) = 5;
    A.at<uint16_t>(5, 0) = 1;
    A.at<uint16_t>(4, 0) = 1;
    A.at<uint16_t>(3, 0) = 1;
    A.at<uint16_t>(2, 0) = 1;
    A.at<uint16_t>(1, 0) = 1;

    std::vector<cv::Vec2d> OUT_lines, actual_lines;
    cv::Vec2d OUT_best_line, actual_best_line;
    actual_lines.push_back(cv::Vec2d(2, 0));
    actual_lines.push_back(cv::Vec2d(0, -CV_PI / 2));
    actual_lines.push_back(cv::Vec2d(-1, -CV_PI / 2));
    actual_lines.push_back(cv::Vec2d(-2, -CV_PI / 2));
    actual_lines.push_back(cv::Vec2d(-3, -CV_PI / 2));
    actual_lines.push_back(cv::Vec2d(-4, -CV_PI / 2));

    actual_best_line = cv::Vec2d(2, 0);

    features::extractLines(A, OUT_lines, OUT_best_line, threshold, d_rho, n_rho, d_theta, min_theta);

    std::sort(OUT_lines.begin(), OUT_lines.end(), [](const cv::Vec2d &lhs, const cv::Vec2d &rhs)
              {
                  if (lhs[1] != rhs[1])
                      return lhs[1] < rhs[1];
                  return lhs[0] < rhs[0];
              });
    std::sort(actual_lines.begin(), actual_lines.end(), [](const cv::Vec2d &lhs, const cv::Vec2d &rhs)
              {
                  if (lhs[1] != rhs[1])
                      return lhs[1] < rhs[1];
                  return lhs[0] < rhs[0];
              });
    ASSERT_EQ(OUT_lines.size(), actual_lines.size()) << "Vectors OUT_lines and actual_lines have different sizes";
    for (size_t i = 0, size = OUT_lines.size(); i < size; i++)
    {
        EXPECT_EQ(OUT_lines[i], actual_lines[i]) << "Vectors OUT_lines and actual_lines are different";
    }
    EXPECT_EQ(OUT_best_line, actual_best_line) << "Vectors OUT_best_line and actual_best_line are different";
}

TEST_F(AccumulateTest, handles_decimal_drho_values)
{
    int n_theta = 1, n_rho = 19, threshold = 0;
    double d_theta = CV_PI / 180, min_theta = -CV_PI / 4, d_rho = 0.5;
    cv::Mat A = cv::Mat::zeros(n_rho, n_theta, CV_16UC1);
    A.at<uint16_t>(12, 0) = 1;
    A.at<uint16_t>(10, 0) = 1;
    A.at<uint16_t>(9, 0) = 1;
    A.at<uint16_t>(8, 0) = 1;
    A.at<uint16_t>(6, 0) = 1;

    std::vector<cv::Vec2d> OUT_lines, actual_lines;
    cv::Vec2d OUT_best_line, actual_best_line;
    actual_lines.push_back(cv::Vec2d(1.5, -CV_PI / 4));
    actual_lines.push_back(cv::Vec2d(0.5, -CV_PI / 4));
    actual_lines.push_back(cv::Vec2d(0, -CV_PI / 4));
    actual_lines.push_back(cv::Vec2d(-0.5, -CV_PI / 4));
    actual_lines.push_back(cv::Vec2d(-1.5, -CV_PI / 4));

    actual_best_line = cv::Vec2d(-1.5, -CV_PI / 4);

 
    features::extractLines(A, OUT_lines, OUT_best_line, threshold, d_rho, n_rho, d_theta, min_theta);

    std::sort(OUT_lines.begin(), OUT_lines.end(), [](const cv::Vec2d &lhs, const cv::Vec2d &rhs)
              {
                  if (lhs[1] != rhs[1])
                      return lhs[1] < rhs[1];
                  return lhs[0] < rhs[0];
              });
    std::sort(actual_lines.begin(), actual_lines.end(), [](const cv::Vec2d &lhs, const cv::Vec2d &rhs)
              {
                  if (lhs[1] != rhs[1])
                      return lhs[1] < rhs[1];
                  return lhs[0] < rhs[0];
              });
    ASSERT_EQ(OUT_lines.size(), actual_lines.size()) << "Vectors OUT_lines and actual_lines have different sizes";
    for (size_t i = 0, size = OUT_lines.size(); i < size; i++)
    {
        EXPECT_EQ(OUT_lines[i], actual_lines[i]) << "Vectors OUT_lines and actual_lines are different";
    }
    EXPECT_EQ(OUT_best_line, actual_best_line) << "Vectors OUT_best_line and actual_best_line are different";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}