#include "../hough_lines.h"
#include <gtest/gtest.h>

class InputTest : public ::testing::Test
{
protected:
    cv::Mat img;
    std::vector<cv::Point> actual_pts;

    void SetUp() override
    {
        img = cv::Mat::zeros(cv::Size(5, 5), CV_8UC1);
        img.forEach<uchar>([](uchar &pixel, const int *position) -> void
                           {
                               if (position[1] == 2)
                                   pixel = 255;
                           });
        actual_pts.push_back(cv::Point(2, 0));
        actual_pts.push_back(cv::Point(2, 1));
        actual_pts.push_back(cv::Point(2, 2));
        actual_pts.push_back(cv::Point(2, 3));
        actual_pts.push_back(cv::Point(2, 4));
    }

    void TearDown() override
    {
    }
};

TEST_F(InputTest, image_as_input)
{
    std::vector<cv::Point> pts;
    int max_x = 0, max_y = 0;
    features::input<cv::Mat>(img, pts, max_x, max_y);
    std::sort(pts.begin(), pts.end(), [](const cv::Point &lhs, const cv::Point &rhs)
              {
                  if (lhs.y != rhs.y)
                      return lhs.y < rhs.y;
                  return lhs.x < rhs.x;
              });
    std::sort(actual_pts.begin(), actual_pts.end(), [](const cv::Point &lhs, const cv::Point &rhs)
              {
                  if (lhs.y != rhs.y)
                      return lhs.y < rhs.y;
                  return lhs.x < rhs.x;
              });

    ASSERT_EQ(pts.size(), actual_pts.size()) << "Vectors pts and actual_pts have different sizes";
    for (size_t i = 0, size = pts.size(); i < size; i++)
    {
        EXPECT_EQ(pts[i], actual_pts[i]) << "Vectors pts and actual_pts differ at index " << i;
    }
    EXPECT_EQ(max_x, 4) << "max_x is not 4";
    EXPECT_EQ(max_y, 4) << "max_y is not 4";
}

TEST_F(InputTest, points_as_input)
{
    std::vector<cv::Point> pts;
    int max_x = 0, max_y = 0;
    features::input<std::vector<cv::Point>>(actual_pts, pts, max_x, max_y);
    std::sort(pts.begin(), pts.end(), [](const cv::Point &lhs, const cv::Point &rhs)
              {
                  if (lhs.y != rhs.y)
                      return lhs.y < rhs.y;
                  return lhs.x < rhs.x;
              });
    std::sort(actual_pts.begin(), actual_pts.end(), [](const cv::Point &lhs, const cv::Point &rhs)
              {
                  if (lhs.y != rhs.y)
                      return lhs.y < rhs.y;
                  return lhs.x < rhs.x;
              });
    ASSERT_EQ(pts.size(), actual_pts.size()) << "Vectors pts and actual_pts have different sizes";
    for (size_t i = 0, size = pts.size(); i < size; i++)
    {
        EXPECT_EQ(pts[i], actual_pts[i]) << "Vectors pts and actual_pts differ at index " << i;
    }
    EXPECT_EQ(max_x, 2) << "max_x is not 2";
    EXPECT_EQ(max_y, 4) << "max_y is not 4";
}

TEST_F(InputTest, invalid_input)
{
    bool caught = false;
    try
    {
        std::vector<cv::Vec2d> vecs;
        std::vector<cv::Point> pts;
        int max_x = 0, max_y = 0;
        features::input(vecs, pts, max_x, max_y);
    }
    catch (const std::exception &e)
    {
        caught = true;
        ASSERT_TRUE(caught) << "invalid_argument exception not caught";
        std::cerr << e.what() << '\n';
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}