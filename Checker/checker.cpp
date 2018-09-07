#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
// 7 is a min number, that gets TRUE in comparasion
bool checkRes(Mat &output, Mat &stdoutput) {
	bool res = true;
	if (output.size() != stdoutput.size()) { return false; }
	else {
		for (int i = 0; i < output.rows; ++i) {
			for (int j = 0; j < output.cols; ++j) {
				if ((output.at<Vec3b>(i, j)[0] - stdoutput.at<Vec3b>(i, j)[0] > 7) ||
					(output.at<Vec3b>(i, j)[1] - stdoutput.at<Vec3b>(i, j)[1] > 7) ||
					(output.at<Vec3b>(i, j)[2] - stdoutput.at<Vec3b>(i, j)[2] > 7)
					) return false;
			}
		}
	}

	return res;
}

//checkers needs only img numer
int main(int argc, char **argv) {
	string sourceName = "../../imgs/source", linoutputName = "../../imgs/linoutput", stdoutputName = "../../imgs/stdoutput",
	       tbboutputName = "../../imgs/tbboutput", ompoutputName = "../../imgs/ompoutput";
	cv::Mat source, linoutput, stdfilter, ompoutput, tbboutput;
	int num = atoi(argv[1]); //num of image
	double sigma = stod(argv[2]); //sigma
	sourceName += std::to_string(num) + ".png";
	linoutputName += std::to_string(num) + ".png";
	stdoutputName += std::to_string(num) + ".png";
	tbboutputName += std::to_string(num) + ".png";
	ompoutputName += std::to_string(num) + ".png";
	
	linoutput = imread(linoutputName, CV_LOAD_IMAGE_COLOR); //loads in BGR
	source = imread(sourceName, CV_LOAD_IMAGE_COLOR);
	ompoutput = imread(ompoutputName, CV_LOAD_IMAGE_COLOR);
	tbboutput = imread(tbboutputName, CV_LOAD_IMAGE_COLOR);
	//imshow("output1", linoutput);
	GaussianBlur(source, stdfilter, cv::Size(3, 3), sigma); //just for check, standart cv filter
	//imshow("stdfilter", stdfilter);
	imwrite(stdoutputName, stdfilter);
	stdfilter = cv::imread(stdoutputName, CV_LOAD_IMAGE_COLOR); //loads in BGR
	//imshow("stdfilter", stdfilter);
	printf(checkRes(linoutput, stdfilter) ? "STD filter and LIN are equal\n" : "STD filter and LIN are NOT equal\n");
	printf(checkRes(ompoutput, stdfilter) ? "STD filter and OMP are equal\n" : "STD filter and OMP are NOT equal\n");
	printf(checkRes(tbboutput, stdfilter) ? "STD filter and TBB are equal\n" : "STD filter and TBB are NOT equal\n");
	cv::waitKey();
	return 0;
}