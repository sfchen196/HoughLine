#include "../hough_lines.h"
#include <yaml-cpp/yaml.h>

void display(const std::string &name, const cv::Mat &img, double fx = 1, double fy = 1, int delay = 0);
void printHelp(std::string program_name);
cv::Mat preprocessing(const cv::Mat &img, YAML::Node config);
void drawLines(cv::Mat img, std::vector<cv::Vec2d> lines, cv::Vec2d max_line, YAML::Node config);

int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        printHelp(argv[0]);
        return -1;
    }

    const auto [img_str, yaml_config] = [&]()
    {
        return std::make_tuple(argv[1], argv[2]);
    }();

    const auto image = cv::imread(img_str);
    const auto config = YAML::LoadFile(yaml_config);

    cv::Mat imCrop = preprocessing(image, config);
    cv::Mat edge_image, imCrop_gray;
    std::vector<std::vector<cv::Point>> contours;

    cv::cvtColor(imCrop, imCrop_gray, cv::COLOR_BGR2GRAY);

    auto t1 = std::chrono::high_resolution_clock::now();

    int flag = config["flag"].as<int>();

    std::vector<cv::Vec2d> lines_from_edges;
    cv::Vec2d max_line_from_edges;

    if (flag)
    {
        cv::Canny(imCrop_gray, edge_image, config["Canny_params"][0].as<double>(), config["Canny_params"][0].as<double>());
        features::HoughLines(edge_image, lines_from_edges, max_line_from_edges, config["HoughLine_params"]["d_rho"].as<double>(),
                             config["HoughLine_params"]["d_theta"].as<double>() / 180 * CV_PI, config["HoughLine_params"]["threshold"].as<int>(),
                             config["HoughLine_params"]["min_theta"].as<double>() / 180 * CV_PI, config["HoughLine_params"]["max_theta"].as<double>() / 180 * CV_PI);
    }
    else
    {
        cv::findContours(imCrop_gray, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
        std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point> &lhs, const std::vector<cv::Point> &rhs)
                  { return cv::contourArea(lhs) > cv::contourArea(rhs); });
        int largest_x = imCrop.cols, largest_y = imCrop.rows;
        contours[0].erase(std::remove_if(
                              contours[0].begin(), contours[0].end(), [=](const cv::Point &pt)
                              { return pt.x > largest_x - 5 || pt.y > largest_y - 5 || pt.x < 5 || pt.y < 5; }),
                          contours[0].end());
        features::HoughLines(contours[0], lines_from_edges, max_line_from_edges, config["HoughLine_params"]["d_rho"].as<double>(),
                             config["HoughLine_params"]["d_theta"].as<double>(), config["HoughLine_params"]["threshold"].as<int>(),
                             config["HoughLine_params"]["min_theta"].as<double>(), config["HoughLine_params"]["max_theta"].as<double>());
    }

    // end timing
    auto t2 = std::chrono::high_resolution_clock::now();
    auto dt_in_ms1 = std::chrono::duration<double>(t2 - t1).count() * 1000;
    std::cout << "Total time: " << dt_in_ms1 << std::endl;

    //#############################################################################################################################################################
    //######################################################## drawing and display ################################################################################
    //#############################################################################################################################################################

    cv::Mat from_edges;
    image.copyTo(from_edges);
    drawLines(from_edges, lines_from_edges, max_line_from_edges, config);

    // display images
    display("lines_from_edges", from_edges, 0.2, 0.2, 0);

    cv::destroyAllWindows();
}

void display(const std::string &name, const cv::Mat &img, double fx, double fy, int delay)
{
    cv::Mat img_vis;
    cv::resize(img, img_vis, cv::Size(), fx, fy);
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, img_vis);
    cv::waitKey(delay);
}
void printHelp(std::string program_name)
{
    std::cout << "ERROR: Could not parse input arguments.\n";
    std::cout << program_name << " <image> <yaml_config> \n";
}

cv::Mat preprocessing(const cv::Mat &img, YAML::Node config)
{
    int fs = config["Gaussian_params"].as<int>();
    cv::GaussianBlur(img, img, cv::Size(fs, fs), 0);
    cv::threshold(img, img, config["Threshold_params"][0].as<int>(),
                  config["Threshold_params"][1].as<int>(), cv::THRESH_BINARY);

    /* 1.2 select ROI 
*/
    cv::Rect roi_rect;
    cv::Point roi_topleft;
    auto node = config["roi"];
    std::vector<int> roi = node.as<std::vector<int>>();
    roi_rect = cv::Rect(roi[0], roi[1], roi[2], roi[3]);
    roi_topleft = roi_rect.tl();
    cv::Mat imCrop = img(roi_rect);
    return imCrop;
}

void drawLines(cv::Mat img, std::vector<cv::Vec2d> lines, cv::Vec2d max_line, YAML::Node config)
{
    /* draw contours and contour from maximum area
    */
    double distance_factor_btw_pts = std::max(config["roi"][2].as<int>(), config["roi"][3].as<int>());
    cv::Point roi_topleft = cv::Point(config["roi"][0].as<int>(), config["roi"][1].as<int>());

    for (auto &params : lines)
    {
        std::vector<cv::Point> pts_in_roi = features::polarLine2cartPoints(params[0], params[1], distance_factor_btw_pts);
        std::vector<cv::Point> pts_in_orig = features::reverseROI(roi_topleft, pts_in_roi);
        cv::line(img, pts_in_orig[0], pts_in_orig[1], cv::Scalar(255, 0, 0), 10);
    }

    /* 4.2 draw the line from highest votes
*/
    std::vector<cv::Point> pts_from_edges = features::reverseROI(roi_topleft, features::polarLine2cartPoints(max_line[0], max_line[1], distance_factor_btw_pts));
    cv::line(img, pts_from_edges[0], pts_from_edges[1], cv::Scalar(0, 0, 255), 10);
    std::cout << "Line_from_edges of highest votes (red): { rho: " << max_line[0] << " theta: "
              << max_line[1] << " }" << std::endl;
}