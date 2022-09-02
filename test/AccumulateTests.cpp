#include "../utils.cpp"
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
    int n_theta = 1, n_rho = 11;
    double angles[1] = {0};
    double d_rho = 1;
    cv::Mat A = cv::Mat::zeros(n_rho, n_theta, CV_16UC1);
    utils::accumulate(margin, angles, n_theta, A, d_rho, n_rho);
    cv::Mat B = cv::Mat::zeros(n_rho, n_theta, CV_16UC1);
    B.at<uint16_t>(7, 0) = 5;
    B.forEach<uint16_t>([&](uint16_t &pixel, const int *position) -> void {
        EXPECT_EQ(A.at<uint16_t>(position[0], position[1]), pixel) << "Accumulator matrix A and B differ at index (" << position[0] << ", " << position[1] << ")";
    });
}

TEST_F(AccumulateTest, two_theta_values)
{
    int n_theta = 2, n_rho = 11;
    double angles[2] = {0, -CV_PI/2};
    double d_rho = 1;
    cv::Mat A = cv::Mat::zeros(n_rho, n_theta, CV_16UC1);
    utils::accumulate(margin, angles, n_theta, A, d_rho, n_rho);
    cv::Mat B = cv::Mat::zeros(n_rho, n_theta, CV_16UC1);
    B.at<uint16_t>(7, 0) = 5;
    B.at<uint16_t>(5, 1) = 1;
    B.at<uint16_t>(4, 1) = 1;
    B.at<uint16_t>(3, 1) = 1;
    B.at<uint16_t>(2, 1) = 1;
    B.at<uint16_t>(1, 1) = 1;
    B.forEach<uint16_t>([&](uint16_t &pixel, const int *position) -> void {
        EXPECT_EQ(A.at<uint16_t>(position[0], position[1]), pixel) << "Accumulator matrix A and B differ at index (" << position[0] << ", " << position[1] << ")";
    });
}

TEST_F(AccumulateTest, handles_decimal_drho_values)
{
    int n_theta = 1, n_rho = 19;
    double angles[1] = {-CV_PI/4};
    double d_rho = 0.5;
    cv::Mat A = cv::Mat::zeros(n_rho, n_theta, CV_16UC1);
    utils::accumulate(margin, angles, n_theta, A, d_rho, n_rho);
    cv::Mat B = cv::Mat::zeros(n_rho, n_theta, CV_16UC1);
    B.at<uint16_t>(12, 0) = 1;
    B.at<uint16_t>(10, 0) = 1;
    B.at<uint16_t>(9, 0) = 1;
    B.at<uint16_t>(8, 0) = 1;
    B.at<uint16_t>(6, 0) = 1;
    B.forEach<uint16_t>([&](uint16_t &pixel, const int *position) -> void {
        EXPECT_EQ(A.at<uint16_t>(position[0], position[1]), pixel) << "Accumulator matrix A and B differ at index (" << position[0] << ", " << position[1] << ")";
    });
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}