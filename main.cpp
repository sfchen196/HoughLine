#include "hough_lines.h"
#include <yaml-cpp/yaml.h>

static void display(const std::string &name, const cv::Mat &img, double fx = 1, double fy = 1, int delay = 0)
{
    cv::Mat img_vis;
    cv::resize(img, img_vis, cv::Size(), fx, fy);
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, img_vis);
    cv::waitKey(delay);
}

int main(int argc, char *argv[])
{
    // start timing
    auto t1 = std::chrono::high_resolution_clock::now();

    // 0.0 load function parameters
    YAML::Node config = YAML::LoadFile("config.yaml");

    /* 0.1 load image
*/
    cv::Mat image;
    if (argc > 1)
        image = cv::imread(argv[1], 1); //load image in BGR scale
    else
        image = cv::imread("../sudoku.jpeg");

    display("image", image, 0.2, 0.2, 1);

    /* 1.1 Preprocessing: 
    a. filter noise with Gaussian blurring
    b. extract white bulk with binary thresholding
*/
    int fs = config["Gaussian_params"].as<int>();
    cv::GaussianBlur(image, image, cv::Size(fs, fs), 0);
    cv::threshold(image, image, config["Threshold_params"][0].as<int>(),
                  config["Threshold_params"][1].as<int>(), cv::THRESH_BINARY);
 
    display("threshold", image, 0.2, 0.2, 1);

    /* 1.2 select ROI 
*/
    cv::Rect roi_rect;
    cv::Point roi_topleft; //used later for mapping the pts in roi to the original image

    auto node = config["roi"];

    std::vector<int> roi = node.as<std::vector<int>>();
    roi_rect = cv::Rect(roi[0], roi[1], roi[2], roi[3]);
    

    roi_topleft = roi_rect.tl();

    cv::Mat imCrop = image(roi_rect);

    display("imCrop", imCrop, 0.5, 0.5, 1);

    /* 2. find edges
*/
    cv::Mat edge_image;
    cv::Mat imCrop_gray;
    cv::cvtColor(imCrop, imCrop_gray, cv::COLOR_BGR2GRAY);
    cv::Canny(imCrop_gray, edge_image, 80, 200);

    display("edges", edge_image, 0.5, 0.5, 1);

    /* 3.1 find contours based on 1) thresholding image, or 2) edge image
    3.2 sort by 1) contour area, or 2) number of pixels in the contour
*/
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(imCrop_gray, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point> &lhs, const std::vector<cv::Point> &rhs)
              { return cv::contourArea(lhs) > cv::contourArea(rhs); });

    /* 3.3 remove parts of the contour at the borders of the image
*/
    int largest_x = imCrop.cols, largest_y = imCrop.rows;

    contours[0].erase(std::remove_if(
                          contours[0].begin(), contours[0].end(), [=](const cv::Point &pt)
                          { return pt.x > largest_x - 5 || pt.y > largest_y - 5 || pt.x < 5 || pt.y < 5; }),
                      contours[0].end());

    /* 4. apply hough line algorithm on both the contour and the edge image
*/
    std::vector<cv::Vec2d> lines_from_edges;
    cv::Vec2d max_line_from_edges;

    // features::HoughLines(edge_image, lines_from_edges, max_line_from_edges, 1, CV_PI / 180, config["HoughLine_params_edges"].as<int>(), -1/5*CV_PI, 1/5*CV_PI);

    features::HoughLines(edge_image, lines_from_edges, max_line_from_edges, 1, CV_PI / 180, config["HoughLine_params_edges"].as<int>(), -0.6*CV_PI, -0.4*CV_PI);

    std::vector<cv::Vec2d> lines_from_contours;
    cv::Vec2d max_line_from_contours;
    features::HoughLines(contours[0], lines_from_contours, max_line_from_contours, 1, CV_PI / 180, config["HoughLine_params_contours"].as<int>(), -0.5*CV_PI, 0.5*CV_PI);

    // end timing
    auto t2 = std::chrono::high_resolution_clock::now();
    auto dt_in_ms1 = std::chrono::duration<double>(t2 - t1).count() * 1000;
    std::cout << "Total time: " << dt_in_ms1 << std::endl;

    //#############################################################################################################################################################
    //######################################################## drawing and display ################################################################################
    //#############################################################################################################################################################

    /* draw contours and contour from maximum area
    */
    double distance_factor_btw_pts = std::max(imCrop.rows, imCrop.cols);

    cv::Mat from_edges;
    image.copyTo(from_edges);

    for (auto &params : lines_from_edges)
    {
        std::vector<cv::Point> pts_in_roi = features::polarLine2cartPoints(params[0], params[1], distance_factor_btw_pts);
        std::vector<cv::Point> pts_in_orig = features::reverseROI(roi_topleft, pts_in_roi);
        std::cout << "rho: " << params[0] << " theta: " << params[1];
        cv::line(from_edges, pts_in_orig[0], pts_in_orig[1], cv::Scalar(255, 0, 0), 10);
    }

    cv::Mat from_contours;
    image.copyTo(from_contours);
    for (auto &params : lines_from_contours)
    {
        std::vector<cv::Point> pts_in_roi = features::polarLine2cartPoints(params[0], params[1], distance_factor_btw_pts);
        std::vector<cv::Point> pts_in_orig = features::reverseROI(roi_topleft, pts_in_roi);
        std::cout << "rho: " << params[0] << " theta: " << params[1];
        cv::line(from_contours, pts_in_orig[0], pts_in_orig[1], cv::Scalar(255, 0, 0), 10);
    }

    /* 4.2 draw the line from highest votes
*/
    std::vector<cv::Point> pts_from_edges = features::reverseROI(roi_topleft, features::polarLine2cartPoints(max_line_from_edges[0], max_line_from_edges[1], distance_factor_btw_pts));
    cv::line(from_edges, pts_from_edges[0], pts_from_edges[1], cv::Scalar(0, 0, 255), 10);
    std::cout << "Line_from_edges of highest votes (red): { rho: " << max_line_from_edges[0] << " theta: "
              << max_line_from_edges[1] << " }" << std::endl;

    std::vector<cv::Point> pts_from_contours = features::reverseROI(roi_topleft, features::polarLine2cartPoints(max_line_from_contours[0], max_line_from_contours[1], distance_factor_btw_pts));
    cv::line(from_contours, pts_from_contours[0], pts_from_contours[1], cv::Scalar(0, 0, 255), 10);
    std::cout << "Line_from_contours of highest votes (red): { rho: " << max_line_from_contours[0] << " theta: "
              << max_line_from_contours[1] << " }" << std::endl;

    // display images
    display("lines_from_edges", from_edges, 0.2, 0.2, 1);

    display("lines_from_contours", from_contours, 0.2, 0.2, 0);


    cv::destroyAllWindows();
}