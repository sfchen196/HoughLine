#include "../utils.cpp"
#include <gtest/gtest.h>

class PrepareAMTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }

    void TearDown() override
    {
    }
};

TEST_F(PrepareAMTest, handles_standard_inputs)
{
    const auto [n_rho, n_theta] = utils::prepareAccumulatorMatrix(2, 4, 1, -CV_PI / 180 * 2, CV_PI / 180 * 2, CV_PI / 180);
    EXPECT_EQ(n_rho, 11) << "n_rho is not 11";
    EXPECT_EQ(n_theta, 5) << "n_theta is not 5";
}

TEST_F(PrepareAMTest, handles_decimal_drho_value)
{
    const auto [n_rho, n_theta] = utils::prepareAccumulatorMatrix(4, 4, 0.5, 0, CV_PI / 180 * 2, CV_PI / 180);
    EXPECT_EQ(n_rho, 25) << "n_rho is not 25";
    EXPECT_EQ(n_theta, 3) << "n_theta is not 3";
}

TEST_F(PrepareAMTest, handles_theta_range)
{
    const auto [n_rho, n_theta] = utils::prepareAccumulatorMatrix(4, 4, 2, 0.1, 0.1, CV_PI / 180);
    EXPECT_EQ(n_rho, 7) << "n_rho is not 7";
    EXPECT_EQ(n_theta, 1) << "n_theta is not 1";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}