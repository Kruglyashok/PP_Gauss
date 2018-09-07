#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "tbb/blocked_range2d.h"
#include "tbb/blocked_range.h"
#include "tbb//task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include <iostream>
#include <cmath>
#include <ctime>
#include <fstream>

//best result with default grainsize(1) intel core i5 6400 2.7ghz

using namespace std;
using namespace cv;
using namespace tbb;
bool typer(Mat& source, string outName)
{
	ofstream file;
	file.open(outName, ofstream::out | ios::binary);
	if (file.is_open())
	{
		if (source.empty())
		{
			int s = 0;
			file.write((char*)(&s), sizeof(int));
			return true;
		}
		int type = source.type();
		file.write((char*)(&source.rows), sizeof(int));
		file.write((char*)(&source.cols), sizeof(int));
		file.write((char*)(&type), sizeof(int));
		file.write((char*)(source.data), source.elemSize() * source.total());
		file.close();
		return true;
	}
	else return false;
}

//fill kernel for gaussian blur
void fillKernel(double **kernel, int radius, double sigma) {
	for (int i = -radius; i <= radius; ++i) {
		for (int j = -radius; j <= radius; ++j) {
			kernel[i + radius][j + radius] = 1 / (2 * CV_PI*sigma*sigma)*(exp(-(i*i + j*j) / (2 * sigma*sigma)));
		}
	}
}
//clamp to prevent negative coordinates
int clamp(int coord, int minValue, int maxValue) {
	int res = coord;
	if (coord < minValue) return minValue;
	if (coord > maxValue) return maxValue;
	else return coord;
	return res;
}

class Gauss {
public:
	Mat& source;
	Mat& output;
	int radius;
	double sigma;
	double** &kernel;
	Gauss(Mat& _source, Mat& _output, int _radius, double _sigma, double** &_kernel) : source(_source), output(_output), radius(_radius), sigma(_sigma), kernel(_kernel) {}
	void operator() (blocked_range2d<int> &r) const {
		for (int x = r.rows().begin(); x != r.rows().end(); ++x) {
			for (int y = r.cols().begin(); y != r.cols().end(); ++y) {
				
				double rSum = 0, gSum = 0, bSum = 0, kSum = 0;
				//work with kernel
				for (int i = 0; i < radius * 2 + 1; ++i) {
					for (int j = 0; j < radius * 2 + 1; ++j) {
						int posX = clamp(x + (i - (radius * 2 + 1) / 2), 0, output.rows - 1);
						int posY = clamp(y + (j - (radius * 2 + 1) / 2), 0, output.cols - 1);
						int r = output.at<Vec3b>(posX, posY)[0];
						int g = output.at<Vec3b>(posX, posY)[1];
						int b = output.at<Vec3b>(posX, posY)[2];
						double kernelVal = kernel[i][j];
						rSum += r * kernelVal;
						gSum += g * kernelVal;
						bSum += b * kernelVal;

						kSum += kernelVal;

					}
				}

				rSum /= kSum;
				gSum /= kSum;
				bSum /= kSum;

				output.at<Vec3b>(x, y)[0] = rSum;
				output.at<Vec3b>(x, y)[1] = gSum;
				output.at<Vec3b>(x, y)[2] = bSum;

			}
		}
	}
};


int main(int argc, char** argv) {
	string sourceName = "../../imgs/source", outputName = "../../imgs/tbboutput";
	int number;
	double** kernel;
	number = atoi(argv[1]);  //1st arg in cmd is the number of the image
	Mat source, output;
	int RADIUS = atoi(argv[3]); // 3d arg is radius; RADIUS ==1  ---> size = 3x3
	sourceName += std::to_string(number) + ".png";
	outputName += std::to_string(number) + ".png";
	
	int numThreads = atoi(argv[4]); //4th arg is numThreads
	try {
		Exception e;
		source = imread(sourceName, CV_LOAD_IMAGE_COLOR);

		if (source.data != NULL) {
			printf("not null\n");
			output = Mat::zeros(source.rows, source.cols, CV_8UC3);
			double sigma = stod(argv[2]);  //2nd arg is sigma
			cout << "sigma = " << sigma << endl;
	double start_time = clock();
		kernel = new double*[RADIUS * 2 + 1];
		for (int i = 0; i < RADIUS * 2 + 1; ++i) kernel[i] = new double[RADIUS * 2 + 1];
		fillKernel(kernel, RADIUS, sigma);
		source.copyTo(output);		
		task_scheduler_init init(numThreads);
		parallel_for(blocked_range2d<int, int>(0, source.rows, source.rows/10, 0 ,source.cols, source.cols/10), Gauss(source, output, RADIUS, sigma, kernel));
		init.~task_scheduler_init();
		delete[]kernel;
	double end_time = clock();
		double filter_time = (end_time - start_time) / CLOCKS_PER_SEC;
			cout << "TBB time = " << filter_time << endl;
			imwrite(outputName, output);
		}
		else {
			printf("null");
			throw e;
		}
	}
	catch (Exception e) {}
	waitKey();
	return 0;
}