#include "../hough_lines.h"
#include <chrono>
// #include <string>
#include <yaml-cpp/yaml.h>

namespace tools
{
    void printHelp(const std::string &program_name);
    void display(const std::string &name, cv::Mat &img, double fx, double fy, int delay);
    cv::Mat drawLines(cv::Mat &img, std::vector<cv::Vec2d> &lines, cv::Vec2d &best_line, const YAML::Node &config, bool drawInCrop);
    cv::Mat drawLines(cv::Mat &img, cv::Vec2d &best_line, const YAML::Node &config, bool drawInCrop);
    void drawContours(std::vector<cv::Point> contour, cv::Mat &dst);
    void ostuBinarization(cv::Mat &input_img, cv::Mat &output_img);
    void eq_hist_with_mask(cv::Mat &src, cv::Mat &dst, cv::Mat &mask);
}
// 1. Preprocessing
//    a. convert to grayscale
//    b. apply binary threshold - try different values
// 2. find edge/contours
//    a. apply Canny edge detection
//    b. apply findContours
// 3. apply 3 different implementations and compare
//
int main(int argc, char **argv)
{
    // if (argc < 3)
    // {
    //     printHelp(argv[0]);
    //     return -1;
    // }
    // const auto [img_str, yaml_config] = [&]()
    // {
    //     return std::make_tuple(argv[1], argv[2]);
    // }();
    // auto image = cv::imread(img_str);
    // auto config = YAML::LoadFile(yaml_config);

    auto config = YAML::LoadFile("../config.yaml");
    size_t num_img;
    std::string bg_color;
    if (config["bg_color"].as<int>() == 1)
    {
        num_img = 5;
        bg_color = "white_cap_";
    }
    else
    {
        num_img = 9;
        bg_color = "black_cap_";
    }
    for (size_t i = 0; i < num_img; i++)
    {
        std::string img_str = "../cap_images/" + bg_color;
        img_str += std::to_string(i) + ".bmp";
        auto image = cv::imread(img_str);
        // 1. Preprocessing
        //    a. select roi - apply dynamic binarization 2 times: (1) isolate the cap; (2) isolate the edges
        //    b. Gaussian filtering and convert to grayscale
        //    c. apply binary threshold - try different values
        cv::Mat image_gray, mask, inverse_mask, bg_filtered, imCrop, imCrop_gray, equalized_tl, equalized_cv, binary_cv, binary_tl, binary;
        if (config["useMask"].as<bool>())
        {
            cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
            int fs = config["Gaussian_params"].as<int>();
            cv::GaussianBlur(image_gray, image_gray, cv::Size(fs, fs), 1);

            tools::ostuBinarization(image_gray, mask);
            cv::bitwise_not(mask, inverse_mask, bg_filtered);

            if (config["bg_color"].as<int>() == 1)
            {
                cv::bitwise_and(image_gray, image_gray, bg_filtered, inverse_mask);

                cv::equalizeHist(bg_filtered, equalized_cv);
                tools::eq_hist_with_mask(image_gray, equalized_tl, inverse_mask);

                tools::ostuBinarization(equalized_cv, binary_cv);
                tools::ostuBinarization(equalized_tl, binary_tl);
                // tools::ostuBinarization(bg_filtered, binary_tl);
                // tools::display("bg_filtered", bg_filtered, 0.5, 0.5, 1);
                // tools::display("binary_tl", binary_tl, 0.5, 0.5, 1);

                binary = binary_tl;
            }
            else
                binary = inverse_mask;
        }
        else
        {
            std::vector<int> roi = config["roi"].as<std::vector<int>>();
            cv::Rect roi_rect = cv::Rect(roi[0], roi[1], roi[2], roi[3]);
            cv::Point roi_topleft = roi_rect.tl();
            imCrop = image(roi_rect);

            // convert to grayscale before or after thresholding ???
            cv::cvtColor(imCrop, imCrop_gray, cv::COLOR_BGR2GRAY);
            int fs = config["Gaussian_params"].as<int>();
            cv::GaussianBlur(imCrop_gray, imCrop_gray, cv::Size(fs, fs), 0);
            cv::threshold(imCrop_gray, binary, config["Threshold_params"][0].as<int>(),
                          config["Threshold_params"][1].as<int>(), cv::THRESH_BINARY);
        } 
        // 2. find edge/contours
        //    a. apply Canny edge detection
        //    b. apply findContours

        cv::Mat edge_image, contour_image;
        std::vector<std::vector<cv::Point>> contours;

        std::vector<cv::Vec2d> OUT_lines;
        cv::Vec2d OUT_best_line;

        if (config["useEdges"].as<bool>())
        {
            cv::Canny(binary, edge_image, config["Canny_params"][0].as<double>(), config["Canny_params"][0].as<double>());
            features::HoughLines(edge_image, OUT_lines, OUT_best_line, config["HoughLine_params"]["d_rho"].as<double>(),
                                 config["HoughLine_params"]["d_theta"].as<double>() / 180 * CV_PI, config["HoughLine_params"]["threshold"].as<int>(),
                                 config["HoughLine_params"]["min_theta"].as<double>() / 180 * CV_PI, config["HoughLine_params"]["max_theta"].as<double>() / 180 * CV_PI);
        }
        else
        {
            cv::findContours(binary, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
            std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point> &lhs, const std::vector<cv::Point> &rhs)
                      { return cv::contourArea(lhs) > cv::contourArea(rhs); });
            int largest_x = binary.cols, largest_y = binary.rows;
            contours[0].erase(std::remove_if(
                                  contours[0].begin(), contours[0].end(), [=](const cv::Point &pt)
                                  { return pt.x > largest_x - 5 || pt.y > largest_y - 5 || pt.x < 5 || pt.y < 5; }),
                              contours[0].end());
            binary.copyTo(contour_image);
            tools::drawContours(contours[1], contour_image);
            tools::display("contour", contour_image, 0.5, 0.5, 1);
            // std::cout << contours[0] << std::endl;
            features::HoughLines(contours[1], OUT_lines, OUT_best_line, config["HoughLine_params"]["d_rho"].as<double>(),
                                 config["HoughLine_params"]["d_theta"].as<double>() / 180 * CV_PI, config["HoughLine_params"]["threshold"].as<int>(),
                                 config["HoughLine_params"]["min_theta"].as<double>() / 180 * CV_PI, config["HoughLine_params"]["max_theta"].as<double>() / 180 * CV_PI);
        }

        //#############################################################################################################################################################
        // ######################################################## drawing and display ################################################################################
        //#############################################################################################################################################################
        cv::Mat img_lines;
        if (config["drawLinesInCrop"].as<bool>())
        {
            if (config["drawBestLineOnly"].as<bool>())
            {
                if (config["useMask"].as<bool>())
                    img_lines = tools::drawLines(image, OUT_best_line, config, true);
                else
                    img_lines = tools::drawLines(imCrop, OUT_best_line, config, true);
            }
            else
            {
                if (config["useMask"].as<bool>())
                    img_lines = tools::drawLines(image, OUT_lines, OUT_best_line, config, true);
                else
                    img_lines = tools::drawLines(imCrop, OUT_lines, OUT_best_line, config, true);
            }
        }
        else
        {
            if (config["drawBestLineOnly"].as<bool>())
            {
                img_lines = tools::drawLines(image, OUT_best_line, config, false);
            }
            else
            {
                img_lines = tools::drawLines(image, OUT_lines, OUT_best_line, config, false);
            }
        }

        // display images
        tools::display("orig", image, 0.5, 0.5, 1);
        // tools::display("mask", mask, 0.5, 0.5, 1);
        // tools::display("inverse_mask", inverse_mask, 0.5, 0.5, 1);
        // tools::display("bg_filtered", bg_filtered, 0.5, 0.5, 1);
        // tools::display("equalized_cv", equalized_cv, 0.5, 0.5, 1);
        // tools::display("equalized_tl", equalized_tl, 0.5, 0.5, 1);
        // tools::display("binary_cv", binary_cv, 0.5, 0.5, 1);
        // tools::display("binary_tl", binary_tl, 0.5, 0.5, 1);
        tools::display("imCrop", imCrop, 3, 3, 1);
        tools::display("imCrop_gray", imCrop_gray, 1, 1, 1);

        tools::display("binary", binary, 0.5, 0.5, 1);
        tools::display("edges", edge_image, 0.5, 0.5, 1);
        tools::display("lines", img_lines, 0.5, 0.5, 0);
        cv::destroyAllWindows();
    }
}

