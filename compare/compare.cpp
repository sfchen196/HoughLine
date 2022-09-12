#include "../hough_lines.h"
#include <chrono>
#include <fstream>
#include <yaml-cpp/yaml.h>

void printHelp(std::string program_name)
{
    std::cout << "ERROR: Could not parse input arguments.\n";
    std::cout << program_name << " <yaml_config> \n";
}
// 1. construct image
// 1a. make a function that takes an image, theta and rho,
// and returns the image with pixels on the line
// 2. compare accuracy
// 2a. apply opencv::houghline and features::houghline on the image
// 2b. compare the output rhos and thetas with the vectors of rhos and thetas
// used in constructing the image
// 3. compare time
// 3a. record time stamps before and after applying the algorithms
// and compare time afterwards
struct Line
{
    double rho;
    double theta;
    double diff(const Line &other) const
    {
        return sqrt(pow((other.rho - rho) / rho, 2) +
                    pow((other.theta - theta) / theta, 2));
    }
};

void addLine(cv::Mat &img, Line &line, double tolerance)
{
    img.forEach<cv::Vec3b>([&](cv::Vec3b &pixel, const int *position) -> void
                           {
                               int x = position[1];
                               int y = position[0];
                               if (abs(y * sin(line.theta) + x * cos(line.theta) - line.rho) < tolerance)
                                   pixel = cv::Vec3b({255, 255, 255});
                           });
}
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printHelp(argv[0]);
        return -1;
    }
    const auto config = YAML::LoadFile(argv[1]);

    std::ofstream fout_acc, fout_time;
    fout_acc.open("/home/sfchen/git/hough-line/compare/compare_acc.txt");
    fout_acc.clear();
    fout_time.open("/home/sfchen/git/hough-line/compare/compare_time.txt");
    fout_time.clear();

    for (double i = 0; i < 90; i++)
    {
        std::cout << i << std::endl;
// set up a binary blank image containing a line of theta = i
        int img_size = config["img_size"].as<int>();
        cv::Mat img = cv::Mat::zeros(img_size, img_size, CV_8UC3);
        double theta = i / 180 * CV_PI;
        double rho = cos(theta) * 250 + sin(theta) * 250;
        Line line = {rho, theta};
        addLine(img, line, config["tolerance"].as<double>());
// applied different algorithm implementations of HoughLines
        cv::Mat img_gray;
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Vec2f> lines_from_cv;
        std::vector<cv::Vec4i> lines_from_cvP;
        std::vector<cv::Vec2d> lines_from_feat;
        cv::Vec2d best_line;

        auto t1 = std::chrono::high_resolution_clock::now();
        cv::HoughLines(img_gray, lines_from_cv, 1, CV_PI / 180, config["threshold_cv"].as<int>(), theta - CV_PI / 180, theta + CV_PI / 180);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        cv::HoughLinesP(img_gray, lines_from_cvP, 1, CV_PI / 180, config["threshold_cvP"].as<int>(), theta - CV_PI / 180, theta + CV_PI / 180);
        auto t3 = std::chrono::high_resolution_clock::now();

        features::HoughLines(img_gray, lines_from_feat, best_line, 1, CV_PI / 180, config["threshold_feat"].as<int>(), theta - CV_PI / 180, theta + CV_PI / 180);
        auto t4 = std::chrono::high_resolution_clock::now();

//#############################################################################################################################################################
//######################################################## drawing and display ################################################################################
//#############################################################################################################################################################
        img.forEach<cv::Vec3b>([&](cv::Vec3b &pixel, const int *position) -> void
                {
                    if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
                        pixel = cv::Vec3b({100,100,100}); 
                });
        cv::Mat img_cv, img_cvP, img_feat;
        img.copyTo(img_cv);
        img.copyTo(img_cvP);
        img.copyTo(img_feat);
        fout_time << "-----------------------" << std::endl;
        fout_time << "angle: " << i << std::endl;
        auto dt_in_ms1 = std::chrono::duration<double>(t2 - t1).count() * 1000;
        fout_time << "Total time - cv: " << dt_in_ms1 << std::endl;
        auto dt_in_ms2 = std::chrono::duration<double>(t3 - t2).count() * 1000;
        fout_time << "Total time - cvP: " << dt_in_ms2 << std::endl;
        auto dt_in_ms3 = std::chrono::duration<double>(t4 - t3).count() * 1000;
        fout_time << "Total time - feat: " << dt_in_ms3 << std::endl;

        fout_acc << "                       " << std::endl;
        fout_acc << "ooooooooooooooooooooooo" << std::endl;
        fout_acc << i << ", ";
        fout_acc << rho << std::endl;

        fout_acc << "ccccccccccccccccccccccc" << std::endl;
        for (size_t j = 0, size = lines_from_cv.size(); j < size; j++)
        {
            fout_acc << lines_from_cv[j][1] / CV_PI * 180 << ", " << lines_from_cv[j][0] << std::endl;
            auto pts = features::polarLine2cartPoints(lines_from_cv[j][0], lines_from_cv[j][1], img_size*2);
            cv::line(img_cv, pts[0], pts[1], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
 
        fout_acc << "PPPPPPPPPPPPPPPPPPPPPPP" << std::endl;
        for (size_t j = 0, size = lines_from_cvP.size(); j < size; j++)
        {
            double x1 = lines_from_cvP[j][0], y1 = lines_from_cvP[j][1], 
                x2 = lines_from_cvP[j][2], y2 = lines_from_cvP[j][3];
            double theta = -1 * atan((x1 - x2) / (y1 - y2));
            double rho = y1 * sin(theta) + x1 * cos(theta);
            fout_acc << theta / CV_PI * 180 << ", " << rho << std::endl;
            auto pts = std::vector<cv::Point>{cv::Point(x1, y1),
                        cv::Point(x2, y2)};
            cv::line(img_cvP, pts[0], pts[1], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }

        fout_acc << "fffffffffffffffffffffff" << std::endl;
        for (size_t j = 0, size = lines_from_feat.size(); j < size; j++)
        {
            fout_acc << lines_from_feat[j][1] / CV_PI * 180 << ", " << lines_from_feat[j][0] << std::endl;
            auto pts = features::polarLine2cartPoints(lines_from_feat[j][0], lines_from_feat[j][1], img_size*2);
            cv::line(img_feat, pts[0], pts[1], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
        fout_acc << "f_best_line" << std::endl;
        fout_acc << best_line[1] / CV_PI * 180 << ", " << best_line[0] << std::endl;

        cv::imshow("img", img);
        cv::waitKey(100);
        cv::imshow("img_cv", img_cv);
        cv::waitKey(100);
        cv::imshow("img_cvP", img_cvP);
        cv::waitKey(100);
        cv::imshow("img_feat", img_feat);
        cv::waitKey(100);
    }
    cv::destroyAllWindows();
    fout_acc.close();
    fout_time.close();
}