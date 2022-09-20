#include "../hough_lines.h"
#include <yaml-cpp/yaml.h>

namespace tools
{

    void printHelp(const std::string &program_name)
    {
        std::cout << "ERROR: Could not parse input arguments.\n";
        std::cout << program_name << " <image> <yaml_config> \n";
    }
    void display(const std::string &name, cv::Mat &img, double fx, double fy, int delay)
    {
        cv::Mat img_vis;
        cv::resize(img, img_vis, cv::Size(), fx, fy);
        cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
        cv::imshow(name, img_vis);
        cv::waitKey(delay);
    }
    void addLine(cv::Mat &img, double theta, double rho, double tolerance)
    {
        img.forEach<cv::Vec3b>([&](cv::Vec3b &pixel, const int *position) -> void
                               {
                               int x = position[1];
                               int y = position[0];
                               if (abs(y * sin(theta) + x * cos(theta) - rho) < tolerance)
                                   pixel = cv::Vec3b({255, 255, 255}); });
    }
    cv::Mat drawLines(cv::Mat &img, std::vector<cv::Vec2d> &lines, cv::Vec2d &best_line, const YAML::Node &config, bool drawInCrop)
    {
        cv::Mat img_lines;
        img.copyTo(img_lines);
        /* draw contours and contour from maximum area
         */
        cv::Scalar lines_color = cv::Scalar(0, 255, 0), best_line_color = cv::Scalar(0, 0, 255);
        int thickness = 1;
        // double distance_factor_btw_pts = std::max(config["roi"][2].as<int>(), config["roi"][3].as<int>());
        double distance_factor_btw_pts = std::max(img.rows, img.cols);
        cv::Point roi_topleft = cv::Point(config["roi"][0].as<int>(), config["roi"][1].as<int>());

        for (auto &params : lines)
        {
            std::vector<cv::Point> pts;
            if (drawInCrop)
            {
                pts = features::polarLine2cartPoints(params[0], params[1], distance_factor_btw_pts);
            }
            else
            {
                std::vector<cv::Point> pts_in_roi = features::polarLine2cartPoints(params[0], params[1], distance_factor_btw_pts);
                pts = features::reverseROI(roi_topleft, pts_in_roi);
            }
            cv::line(img_lines, pts[0], pts[1], lines_color, thickness, cv::LINE_AA);
        }

        /* 4.2 draw the line from highest votes
         */
        std::vector<cv::Point> pts;
        if (drawInCrop)
        {
            pts = features::polarLine2cartPoints(best_line[0], best_line[1], distance_factor_btw_pts);
        }
        else
        {
            std::vector<cv::Point> pts_in_roi = features::polarLine2cartPoints(best_line[0], best_line[1], distance_factor_btw_pts);
            pts = features::reverseROI(roi_topleft, pts_in_roi);
        }
        cv::line(img_lines, pts[0], pts[1], best_line_color, thickness, cv::LINE_AA);
        std::cout << "Line of highest votes (red): { rho: " << best_line[0] << " theta: "
                  << best_line[1] / CV_PI * 180 << " }" << std::endl;
        return img_lines;
    }
    cv::Mat drawLines(cv::Mat &img, cv::Vec2d &best_line, const YAML::Node &config, bool drawInCrop)
    {
        auto lines = std::vector<cv::Vec2d>();
        return drawLines(img, lines, best_line, config, drawInCrop);
    }
    void drawContours(std::vector<cv::Point> contour, cv::Mat &dst)
    {
        dst = cv::Mat::zeros(dst.rows, dst.cols, CV_8UC1);
        for (const auto &pt : contour)
        {
            dst.at<uchar>(pt.y, pt.x) = 255;
        }
    }
    // the method only focuses on the non-zero part of the input_img
    void ostuBinarization(cv::Mat &input_img, cv::Mat &output_img)
    {
        double histogram[256] = {0};
        double min_value = 255;
        double max_value = 0;
        double aver_value = 0;
        double sum = 0;
        for (int j = 0; j < input_img.rows; ++j)
        {
            for (int i = 0; i < input_img.cols; ++i)
            {
                int value = input_img.at<uchar>(j, i);
                // if (value == 0)
                // if (value == 255)
                //     continue;
                histogram[value]++;
                if (value < min_value)
                    min_value = value;
                if (value > max_value)
                    max_value = value;
                aver_value += value;
                sum++;
            }
        }
        aver_value = aver_value / sum;
        double thresh;
        double final_thresh = 128;
        double max_variance = 0;
        for (thresh = min_value; thresh < max_value; ++thresh)
        {
            double sum_num_ob = 0, sum_num_bg = 0;
            double sum_value_ob = 0, sum_value_bg = 0;
            for (unsigned int h = 0; h < 256; ++h)
            {
                if (h > thresh)
                {
                    sum_num_ob += histogram[h];
                    sum_value_ob += histogram[h] * h;
                }
                else
                {
                    sum_num_bg += histogram[h];
                    sum_value_bg += histogram[h] * h;
                }
            }
            double variance = sum_num_ob / sum * std::pow(sum_value_ob / sum_num_ob - aver_value, 2) +
                              sum_num_bg / sum * std::pow(sum_value_bg / sum_num_bg - aver_value, 2);
            if (max_variance < variance)
            {
                max_variance = variance;
                final_thresh = thresh;
            }
        }
        std::cout << "threshold: " << final_thresh << std::endl;
        cv::threshold(input_img, output_img, final_thresh, 255, 0);
    }
    void eq_hist_with_mask(cv::Mat &src, cv::Mat &dst, cv::Mat &mask)
    {
        // TODO: need to check if src is 8-bit single-channel image
        cv::Mat masked_src;
        dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
        cv::bitwise_and(src, src, masked_src, mask);
        // 1. calculate histogram
        size_t hist[256] = {0};
        size_t total = 0;
        for (size_t i = 0; i < masked_src.rows; i++)
        {
            for (size_t j = 0; j < masked_src.cols; j++)
            {
                uchar pixel = masked_src.at<uchar>(i, j);
                if (pixel)
                {
                    total++;
                    hist[pixel]++;
                }
            }
        }
        // 2. calculate cumulative histogram
        size_t cumulative_hist[256] = {0};
        size_t curr_sum = 0;
        for (size_t i = 0; i < 256; i++)
        {
            // cumulative_hist[i] = hist[i] + cumulative_hist[i-1];
            cumulative_hist[i] = hist[i] + curr_sum;
            curr_sum = cumulative_hist[i];
        }
        // 3. calculate output pixel intensity
        masked_src.forEach<uchar>([&] (uchar &pixel, const int* position) -> void
        {
                if (pixel)
                {
                    uchar intensity = (uchar) round(255 * (1.0 * cumulative_hist[pixel] / total));
                    dst.at<uchar>(position[0], position[1]) = intensity;
                }
                else
                {
                    // dst.at<uchar>(position[0], position[1]) = src.at<uchar>(position[0], position[1]);
                    dst.at<uchar>(position[0], position[1]) = pixel;
                }
        });
    }
}
