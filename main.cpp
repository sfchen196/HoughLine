#include <yaml-cpp/yaml.h>
#include "hough_lines.h"


static void display(const std::string &name, cv::Mat &img, double fx = 1, double fy = 1, int delay = 0)
{
    cv::Mat img_vis;
    cv::resize(img, img_vis, cv::Size(), fx, fy);
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, img_vis);
    cv::waitKey(delay);
}

static void parametric_line(cv::Mat &img, double rho, double theta, const cv::Scalar &color = cv::Scalar(0, 0, 255),
                            int thickness = 1, int lineType = cv::LINE_AA)
{
    double x0 = rho * cos(theta); // the x-coordinate of the intersection pt between the line itself and the perpendicular line where rho lands.
    double y0 = rho * sin(theta); // the y-coordinate of the intersection pt between the line itself and the perpendicular line where rho lands.
    double factor = std::max(img.rows, img.cols);
    cv::Point p1(x0 + factor * (-sin(theta)), y0 + factor * (cos(theta))); // from the intersection point (x0, y0), go along the line leftward 
    cv::Point p2(x0 - factor * (-sin(theta)), y0 - factor * (cos(theta))); // from the intersection point (x0, y0), go along the line rightward
    cv::line(img, p1, p2, color, thickness, lineType);
}

static std::vector<cv::Point> reverse_roi(cv::Point &roi_topleft, std::vector<cv::Point> &points)
{
    std::vector<cv::Point> pts = std::vector<cv::Point>(points);

    std::for_each(pts.begin(), pts.end(), [&](cv::Point &pt) {
        pt.x += roi_topleft.x;
        pt.y += roi_topleft.y;
    });
    return pts;
}

int main(int argc, char *argv[])
{
    // start timing
    auto t1 = std::chrono::high_resolution_clock::now();

    // 0.0 load function parameters
    YAML::Node config = YAML::LoadFile("../config.yaml");

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
    int fs = config["Gaussian_filter_size"].as<int>();
    cv::GaussianBlur(image, image, cv::Size(fs, fs), 0);
    cv::threshold(image, image, config["Threshold_params"][0].as<int>(),
                  config["Threshold_params"][1].as<int>(), cv::THRESH_BINARY);
    display("threshold", image, 0.2, 0.2, 1);

    /* 1.2 select ROI 
*/
    cv::Rect2d roi(config["roi"][0].as<int>(), config["roi"][1].as<int>(),
                   config["roi"][2].as<int>(), config["roi"][3].as<int>());
    cv::Mat imCrop = image(roi);
    display("imCrop", imCrop, 1, 0.5, 1);

    /* 2. find edges
*/
    cv::Mat edge_image;
    cv::Mat imCrop_gray;
    cv::cvtColor(imCrop, imCrop_gray, cv::COLOR_BGR2GRAY);
    cv::Canny(imCrop_gray, edge_image, 80, 200);
    display("edges", edge_image, 1, 0.5, 1);

    /* 3.1 find contours based on 1) thresholding image, or 2) edge image
    3.2 sort by 1) contour area, or 2) number of pixels in the contour
*/
    std::vector<std::vector<cv::Point>> contours;
    int flag = config["Contour_params"].as<int>();
    if (flag == 0)
    {
        cv::findContours(imCrop_gray, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
        std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point> &lhs, const std::vector<cv::Point> &rhs)
                  { return cv::contourArea(lhs) > cv::contourArea(rhs); });
    }
    else if (flag == 1)
    {
        cv::findContours(edge_image, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
        std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point> &lhs, const std::vector<cv::Point> &rhs)
                  { return lhs.size() > rhs.size(); });
    }


    /* 3.3 remove parts of the contour at the borders of the image
*/
    int largest_x = imCrop.cols, largest_y = imCrop.rows;
    contours[0].erase(std::remove_if(
                          contours[0].begin(), contours[0].end(), [=](const cv::Point &pt)
                          { return pt.x > largest_x - 5 || pt.y > largest_y - 5 || pt.x < 5 || pt.y < 5; }),
                      contours[0].end());

    /* 4. apply hough line algorithm on both the contour and the edge image
*/
    // std::vector<cv::Vec2d> lines_from_edges;
    // cv::Vec2d max_line_from_edges;
    std::vector<std::vector<cv::Point>> lines_from_edges;
    std::vector<cv::Point> max_line_from_edges;
    features::hough_line(edge_image, lines_from_edges, max_line_from_edges, 1, CV_PI / 180, 100, 0, CV_PI);

    // std::vector<cv::Vec2d> lines_from_contours;
    // cv::Vec2d max_line_from_contours;
    std::vector<std::vector<cv::Point>> lines_from_contours;
    std::vector<cv::Point> max_line_from_contours;
    features::hough_line(edge_image, contours[0], lines_from_contours, max_line_from_contours, 1, CV_PI / 180, 100, 0, CV_PI);

    // end timing
    auto t2 = std::chrono::high_resolution_clock::now();
    auto dt_in_ms1 = std::chrono::duration<double>(t2 - t1).count() * 1000;
    std::cout << "Total time: " << dt_in_ms1 << std::endl;

//#############################################################################################################################################################
//######################################################## drawing and display ################################################################################
//#############################################################################################################################################################

    /* draw contours and contour from maximum area
    */

    cv::RNG rng(12345);
    cv::Mat drawing = cv::Mat::zeros(imCrop.size(), CV_8UC3);
    for (size_t i = 1; i < contours.size(); i++)
    {
        {
            cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), 0);
            cv::drawContours(drawing, contours, (int)i, color, 0.5, cv::LINE_8);
        }
    }


    // cv::drawContours(drawing, contours, 0, cv::Scalar(0, 0, 255), 1, cv::LINE_8);
    for (const auto &pt : contours[0])
        drawing.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(0, 0, 255);
    display("contours", drawing, 1, 0.5, 1);

    /* 4.1 draw lines above thresholds
*/
    cv::Point roi_topleft(config["roi"][0].as<int>(), config["roi"][1].as<int>());
    cv::Mat from_edges;
    image.copyTo(from_edges);
    for (auto &params : lines_from_edges)
    {
        // parametric_line(from_edges, params[0], params[1], cv::Scalar(255, 0, 0));

        std::vector<cv::Point> pts = reverse_roi(roi_topleft, params);
        cv::line(from_edges, pts[0], pts[1], cv::Scalar(255, 0, 0), 10);
    }


    cv::Mat contour_image = cv::Mat::zeros(imCrop.size(), CV_8UC3);
    for (const auto &pt : contours[0])
        contour_image.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(255, 255, 255);

    cv::Mat from_contours;
    image.copyTo(from_contours);
    // imCrop.copyTo(from_contours_roi);
    for (auto &params : lines_from_contours)
    {
        // parametric_line(from_contours, params[0], params[1], cv::Scalar(255, 0, 0));
        std::vector<cv::Point> pts = reverse_roi(roi_topleft, params);
        cv::line(from_contours, pts[0], pts[1], cv::Scalar(255, 0, 0), 10);
    }

    /* 4.2 draw the line from highest votes
*/
    // parametric_line(from_edges, max_line_from_edges[0], max_line_from_edges[1], cv::Scalar(0, 0, 255), 2);
    cv::line(from_edges, reverse_roi(roi_topleft, max_line_from_edges)[0], reverse_roi(roi_topleft, max_line_from_edges)[1], cv::Scalar(0, 0, 255), 10);
    // std::cout << "Line_from_edges of highest votes (red): { pt1: " << line_edges[0] << " pt2: " << line_edges[1] << " }" << std::endl;

    cv::line(from_contours, reverse_roi(roi_topleft, max_line_from_contours)[0], reverse_roi(roi_topleft, max_line_from_contours)[1], cv::Scalar(0, 0, 255), 10);
    // cv::line(contour_image, max_line_from_contours[0], max_line_from_contours[1], cv::Scalar(0, 0, 255), 2);
    // std::cout << "Line_from_contours of highest votes (red): { pt1: " << line_contours[0] << " pt2: " << line_contours[1] << " }" << std::endl;

    // display images
    display("lines_from_edges", from_edges, 0.2, 0.2, 1);
    display("lines_from_contours", from_contours, 0.2, 0.2, 0);
    // display("lines_from_contours", contour_image, 1, 0.5, 0);

    cv::destroyAllWindows();
}