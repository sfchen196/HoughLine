#include "../hough_lines.h"
#include <yaml-cpp/yaml.h>

void printHelp(std::string program_name)
{
    std::cout << "ERROR: Could not parse input arguments.\n";
    std::cout << program_name << " <image> <yaml_config> \n";
}

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

    cv::Mat gray;
    cv::GaussianBlur(image, image, cv::Size(config["Gaussian_params"].as<int>(), config["Gaussian_params"].as<int>()), 0);
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, gray, config["Threshold_params"][0].as<double>(), config["Threshold_params"][1].as<double>(), cv::THRESH_BINARY);
    cv::Mat edge_image;
    cv::Canny(gray, edge_image, config["Canny_params"][0].as<double>(), config["Canny_params"][0].as<double>());


    std::vector<cv::Vec2d> OUT_lines;
    cv::Vec2d OUT_best_line;
    features::HoughLines(edge_image, OUT_lines, OUT_best_line, config["HoughLine_params"]["d_rho"].as<double>(),
                         config["HoughLine_params"]["d_theta"].as<double>() / 180 * CV_PI, config["HoughLine_params"]["threshold"].as<int>(),
                         config["HoughLine_params"]["min_theta"].as<double>() / 180 * CV_PI, config["HoughLine_params"]["max_theta"].as<double>() / 180 * CV_PI);


    for (auto &params : OUT_lines)
    {
        cv::line(image, features::polarLine2cartPoints(params[0], params[1])[0], features::polarLine2cartPoints(params[0], params[1])[1],
                 cv::Scalar(255, 0, 0), 2);
    }
    cv::line(image, features::polarLine2cartPoints(OUT_best_line[0], OUT_best_line[1])[0], features::polarLine2cartPoints(OUT_best_line[0], OUT_best_line[1])[1],
             cv::Scalar(0, 0, 255), 2);
    std::cout << "Line_from_edges of highest votes (red): { rho: " << OUT_best_line[0] << " theta: "
              << OUT_best_line[1] << " }" << std::endl;

    cv::imshow("lines_from_edges", image);
    cv::waitKey(0);

    cv::destroyAllWindows();
}
