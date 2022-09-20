#include "../hough_lines.h"
#include <chrono>
#include <fstream>
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
// 2. find edge/contours
// 3. apply HoughCircles to get the position of the cap
// 4. use the circle position to construct a mask image
// 5. use the mask image and bitwise manipulation to isolate the cap pixels
// 6. use histogram equalization to enhance contrast
// 7. apply dynamic binarization or hardcoding threshold to make it an binary image
// 8. find edge/contours
// 9. apply HoughLines to get the orientation o the cap
int main(int argc, char **argv)
{


    auto config = YAML::LoadFile("../config_cl.yaml");
    size_t num_img;
    std::string cap_color;
    std::ofstream fout;
    if (config["bg_color"].as<int>() == 1)
    {
        num_img = 5;
        cap_color = "white_cap_";
        fout.open("../Time_and_Results_white.txt");
    }
    else
    {
        num_img = 9;
        cap_color = "black_cap_";
        fout.open("../Time_and_Results_black.txt");
    }
    fout.clear();
    fout << "opencv version: " << CV_VERSION << std::endl;
    fout << "Major version: " << CV_MAJOR_VERSION << std::endl;
    fout << "Minor version: " << CV_MINOR_VERSION << std::endl;
    fout << "Subminor version: " << CV_SUBMINOR_VERSION << std::endl;
    fout << std::endl;

    for (size_t i = 0; i < num_img; i++)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::string img_str = "../cap_images/" + cap_color;
        img_str += std::to_string(i) + ".bmp";
        auto image = cv::imread(img_str);
   
        // 1. Preprocessing
        //    a. select roi - apply dynamic binarization 2 times: (1) isolate the cap; (2) isolate the edges
        //    b. Gaussian filtering and convert to grayscale
        //    c. apply binary threshold - try different values
        cv::Mat image_gray, imCrop, imCrop_gray, binary_all, binary_ob, mask, bg_removed, equalized;
        std::vector<int> roi = config["roi"].as<std::vector<int>>();
        cv::Rect roi_rect = cv::Rect(roi[0], roi[1], roi[2], roi[3]);
        cv::Point roi_topleft = roi_rect.tl();
        imCrop = image(roi_rect);

        // convert to grayscale before or after thresholding ???
        cv::cvtColor(imCrop, imCrop_gray, cv::COLOR_BGR2GRAY);
        int fs = config["Gaussian_params"].as<int>();
        cv::GaussianBlur(imCrop_gray, imCrop_gray, cv::Size(fs, fs), 0);

    auto t2 = std::chrono::high_resolution_clock::now();
        if (config["useDynamicThreshold"].as<bool>())
        {
            tools::ostuBinarization(imCrop_gray, binary_all);
        }
        else
        {
            int bg_color = config["bg_color"].as<int>();
            cv::threshold(imCrop_gray, binary_all, config["Threshold_params_circle"][bg_color].as<int>(),
                          config["Threshold_params_circle"][2].as<int>(), cv::THRESH_BINARY);
        }
    auto t3 = std::chrono::high_resolution_clock::now();

        // 2. find edge/contours
        //    a. apply Canny edge detection
        //    b. apply findContours

        cv::Mat edge_image, contour_image;
        std::vector<std::vector<cv::Point>> contours;

        std::vector<cv::Vec3f> OUT_circles;
        cv::HoughCircles(binary_all, OUT_circles, cv::HOUGH_GRADIENT,
                         config["HoughCircle_params"]["dp"].as<double>(),
                         config["HoughCircle_params"]["minDist"].as<double>(),
                         config["HoughCircle_params"]["param1"].as<double>(),
                         config["HoughCircle_params"]["param2"].as<double>(),
                         config["HoughCircle_params"]["minRadius"].as<int>(),
                         config["HoughCircle_params"]["maxRadius"].as<int>());
    auto t3i = std::chrono::high_resolution_clock::now();
        cv::Point circle_center = cv::Point(OUT_circles[0][0], OUT_circles[0][1]);
        int radius = (int) OUT_circles[0][2];
        // construct a mask based on the detected circle
        mask = cv::Mat::zeros(imCrop.rows, imCrop.cols, CV_8UC1);
        mask.forEach<uchar>([&](uchar &pixel, const int* position) -> void {
           double distance = sqrt(pow(position[1] - circle_center.x, 2) + pow(position[0] - circle_center.y, 2));
           if (distance < radius)
           {
               pixel = 255;
           }
        });
        cv::bitwise_and(imCrop_gray, imCrop_gray, bg_removed, mask);
        cv::equalizeHist(bg_removed, equalized);
    auto t4 = std::chrono::high_resolution_clock::now();
        if (config["useDynamicThreshold"].as<bool>())
        {
            tools::ostuBinarization(equalized, binary_ob);
        }
        {
            int bg_color = config["bg_color"].as<int>();
            cv::threshold(equalized, binary_ob, config["Threshold_params_line"][bg_color].as<int>(),
                          config["Threshold_params_line"][2].as<int>(), cv::THRESH_BINARY);
        }

        std::vector<cv::Vec2d> OUT_lines;
        cv::Vec2d OUT_best_line;
        std::vector<cv::Vec3f> OUT_lines_cv;
        std::vector<cv::Vec4f> OUT_lines_cvP;

    auto t5 = std::chrono::high_resolution_clock::now();
        cv::Canny(binary_ob, edge_image, config["Canny_params"][0].as<double>(), config["Canny_params"][0].as<double>());
        if (config["whichHoughLine"].as<int>() == 0)
        {
            cv::HoughLines(edge_image, OUT_lines_cv,
                           config["HoughLine_params"]["d_rho"].as<double>(),
                           config["HoughLine_params"]["d_theta"].as<double>() / 180 * CV_PI,
                           config["HoughLine_params"]["threshold"].as<int>(),
                           0, 0,
                           config["HoughLine_params"]["min_theta"].as<double>() / 180 * CV_PI,
                           config["HoughLine_params"]["max_theta"].as<double>() / 180 * CV_PI);
        }
        else if (config["whichHoughLine"].as<int>() == 1)
        {
            cv::HoughLinesP(edge_image, OUT_lines_cvP,
                            config["HoughLineP_params"]["d_rho"].as<double>(),
                            config["HoughLineP_params"]["d_theta"].as<double>() / 180 * CV_PI,
                            config["HoughLineP_params"]["threshold"].as<int>(),
                            config["HoughLineP_params"]["minLineLength"].as<double>(),
                            config["HoughLineP_params"]["maxLineGap"].as<double>()
                            );
        }
        else
        {
            features::HoughLines(edge_image, OUT_lines, OUT_best_line,
                            config["HoughLine_params"]["d_rho"].as<double>(),
                            config["HoughLine_params"]["d_theta"].as<double>() / 180 * CV_PI,
                            config["HoughLine_params"]["threshold"].as<int>(),
                            config["HoughLine_params"]["min_theta"].as<double>() / 180 * CV_PI,
                            config["HoughLine_params"]["max_theta"].as<double>() / 180 * CV_PI);
        }
    auto t6 = std::chrono::high_resolution_clock::now();
        
        //#############################################################################################################################################################
        // ######################################################## drawing and display ################################################################################
        //#############################################################################################################################################################
        cv::Mat img_circles;
        cv::Mat img_lines;
        imCrop.copyTo(img_circles);
        // auto pts = features::reverseROI(roi_topleft, std::vector<cv::Point>{cv::Point(c[0],c[1])});
        // cv::circle(img_circles, pts[0], 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        // cv::circle(img_circles, pts[0], (int) c[2], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::circle(img_circles, circle_center, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::circle(img_circles, circle_center, radius, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        if (config["whichHoughLine"].as<int>() == 0)
        {
            for (const auto& line : OUT_lines_cv)
            {
                OUT_lines.push_back(cv::Vec2d{ line[0], line[1] });
            }
            OUT_best_line = cv::Vec2d{ OUT_lines_cv[0][0], OUT_lines_cv[0][1] };
        }
        if (config["whichHoughLine"].as<int>() == 1)
        {
            for (const auto& pts : OUT_lines_cvP)
            {
                // imCrop.copyTo(img_lines);
                // cv::line(img_lines,
                //     cv::Point(pts[0], pts[1]),
                //     cv::Point(pts[2], pts[3]), 
                //     cv::Scalar(0, 255, 0));

                auto polar_line = features::cartPoints2polarLine(
                    cv::Point(pts[0], pts[1]), 
                    cv::Point(pts[2], pts[3]));
                OUT_lines.push_back(polar_line);
            }

            // cv::line(img_lines,
            //          cv::Point(OUT_lines_cvP[0][0], OUT_lines_cvP[0][1]),
            //          cv::Point(OUT_lines_cvP[0][2], OUT_lines_cvP[0][3]),
            //          cv::Scalar(0, 0, 255));

            OUT_best_line = features::cartPoints2polarLine(
                cv::Point(OUT_lines_cvP[0][0], OUT_lines_cvP[0][1]),
                cv::Point(OUT_lines_cvP[0][2], OUT_lines_cvP[0][3])
            );
        }

        // if (config["whichHoughLine"].as<int>() != 1)
        if (config["whichHoughLine"].as<int>() < 3)
        {

            if (config["drawBestLineOnly"].as<bool>())
            {
                if (config["drawLinesInCrop"].as<bool>())
                {
                    img_lines = tools::drawLines(imCrop, OUT_best_line, config, true);
                }
                else
                {
                    img_lines = tools::drawLines(image, OUT_best_line, config, false);
                }
            }
            else
            {
                if (config["drawLinesInCrop"].as<bool>())
                {
                    img_lines = tools::drawLines(imCrop, OUT_lines, OUT_best_line, config, true);
                }
                else
                {
                    img_lines = tools::drawLines(image, OUT_lines, OUT_best_line, config, false);
                }
            }
        }

    // display images
        tools::display("orig", image, 0.25, 0.25, 1);
        // tools::display("imCrop", imCrop, 1, 1, 1);
        // tools::display("imCrop_gray", imCrop_gray, 1, 1, 1);

        tools::display("binary_all", binary_all, 1, 1, 1);
        tools::display("circles", img_circles, 1, 1, 1);
        // tools::display("mask", mask, 1, 1, 1);
        // tools::display("bg_removed", bg_removed, 1, 1, 1);
        tools::display("equalized", equalized, 1, 1, 1);
        tools::display("binary_ob", binary_ob, 1, 1, 1);

        // tools::display("edges", edge_image, 1, 1, 1);
        tools::display("lines", img_lines, 1, 1, 0);
        cv::destroyAllWindows();

    // write results to files
        fout << "--------------------------------------------------------------------------------------------" << std::endl;
        fout << "image location: " << img_str << std::endl; 
        fout << "radius: " << radius << std::endl;
        fout << std::endl;

    auto dt_12 = std::chrono::duration<double>(t2 - t1).count() * 1000;
    fout << "preprocessing time: " << dt_12 << std::endl;

    auto dt_23 = std::chrono::duration<double>(t3 - t2).count() * 1000;
    fout << "circle binarization time: " << dt_23 << std::endl;

    auto dt_33i = std::chrono::duration<double>(t3i - t3).count() * 1000;
    fout << "HoughCircle: " << dt_33i << std::endl;
    auto dt_3i4 = std::chrono::duration<double>(t4 - t3i).count() * 1000;
    fout << "mask and hist equalization: " << dt_3i4 << std::endl;

    auto dt_45 = std::chrono::duration<double>(t5 - t4).count() * 1000;
    fout << "line binarization time: " << dt_45 << std::endl;

    auto dt_56 = std::chrono::duration<double>(t6 - t5).count() * 1000;
    fout << "HoughLine: " << dt_56 << std::endl;

    auto dt_16 = std::chrono::duration<double>(t6 - t1).count() * 1000;
    fout << "Total time: " << dt_16 << std::endl;
    }
}
// 1. must use different sets of thresholds for different backgound colors
    // ps. dynamic binarization is doing a good job finding the appropriate threshold, but sometimes gives less optimized values

// 2. not sure if black background itself leads to good detections or it's because that 
// the caps w/ black background are better positioned under the camera
