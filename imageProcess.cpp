#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "opencv/highgui.h"
/* using namespace cv; */

int main() {
  cv::Mat img = cv::imread("images/hoge.png", cv::IMREAD_UNCHANGED);
  cv::rectangle(img, cv::Point(10, 20), cv::Point(50, 60), cv::Scalar(255, 0, 0), 5, 8);
  cv::imshow("out", img);
  cv::waitKey();
  return 0;
}
