#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <string>

using namespace cv;
using namespace std;

int main() {
	string outputName1 = "../../imgs/source", outputName2 = "../../imgs/source", outputName3 = "../../imgs/source";
	outputName1 += std::to_string(6) + ".png";
	outputName2 += std::to_string(7) + ".png";
	outputName3 += std::to_string(8) + ".png";
	Mat out1 = Mat::zeros(256, 256, CV_8UC3), out2 = Mat::ones(256, 256, CV_8UC1) * 255, out3 = Mat::ones(1, 1, CV_8UC1) * 255;
	imshow("1", out1);
	imshow("2", out2);
	imshow("3", out3);
	imwrite(outputName1, out1);
	imwrite(outputName2, out2);
	imwrite(outputName3, out3);
	waitKey();

	return 0;
}