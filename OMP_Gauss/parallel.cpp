#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cmath>
#include <ctime>
#include "omp.h"
#include <fstream>

using namespace cv;
using namespace std;

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

//sequential gaussian filter
void gaussFilter(Mat& source, Mat& output1, int radius, double sigma, int numThreads) {
	if (sigma < 0) return;
	if (source.cols == 1 || source.rows == 1) return;
	if (source.data == NULL) return;
	double **kernel;

	//output1 is an output matrix, we work here
	source.copyTo(output1);
	kernel = new double*[radius * 2 + 1];
	for (int i = 0; i < radius * 2 + 1; ++i) kernel[i] = new double[radius * 2 + 1];
	fillKernel(kernel, radius, sigma);

	//check kernel
	/*
	for (int i = 0; i < radius * 2 + 1; ++i) {
		for (int j = 0; j < radius * 2 + 1; ++j) {
			printf("%10.6f\t", kernel[i][j]);
		}
		std::cout << std::endl;
	}*/


	//work with inner image
	int n = numThreads;
	int tid;
	omp_set_num_threads(n);
	int chunk = source.rows / n;
#pragma omp parallel private(tid)  shared(output1)
	{
		tid = omp_get_thread_num();
		if (tid == 0) {
			cout << "threads: " << n << endl;
		}
#pragma omp for schedule(guided)
		for (int x = 0; x < output1.rows; ++x) {
			for (int y = 0; y < output1.cols; ++y) {
				double rSum = 0, gSum = 0, bSum = 0, kSum = 0;
				//work with kernel
				for (int i = 0; i < radius * 2 + 1; ++i) {
					for (int j = 0; j < radius * 2 + 1; ++j) {
						int posX = clamp(x + (i - (radius * 2 + 1) / 2), 0, output1.rows - 1);
						int posY = clamp(y + (j - (radius * 2 + 1) / 2), 0, output1.cols - 1);
						int r = output1.at<Vec3b>(posX, posY)[0];
						int g = output1.at<Vec3b>(posX, posY)[1];
						int b = output1.at<Vec3b>(posX, posY)[2];
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

				output1.at<Vec3b>(x, y)[0] = rSum;
				output1.at<Vec3b>(x, y)[1] = gSum;
				output1.at<Vec3b>(x, y)[2] = bSum;

			}
		}
	}
	delete[]kernel;
}


int main(int argc, char** argv) {
	string sourceName = "../../imgs/source", outputName = "../../imgs/ompoutput";
	int number;
	number = atoi(argv[1]);  //1st arg in cmd is the number of the image
	Mat source, output;
	int RADIUS = atoi(argv[3]); // 3d arg is radius; RADIUS ==1  ---> size = 3x3
	int numThreads = atoi(argv[4]); //4th arg is numThreads
	sourceName += std::to_string(number) + ".png";
	outputName += std::to_string(number) + ".png";
	try {
		Exception e;
		source = imread(sourceName, CV_LOAD_IMAGE_COLOR);

		if (source.data != NULL) {
			printf("not null\n");
			output = Mat::zeros(source.rows, source.cols, CV_8UC3);
			double sigma = stod(argv[2]);  //2nd arg is sigma
			cout << "sigma = " << sigma << endl;
			double start_time = clock();
			gaussFilter(source, output, RADIUS, sigma, numThreads);
			double end_time = clock();
			double filter_time = (end_time - start_time) / CLOCKS_PER_SEC;
			cout << "OMP time = " << filter_time << endl;
			//imshow("source1", source);
			//imshow("filtered", output);
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