#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
int main()
{
	Mat source,grey;
	source=imread("C:\\Users\\朱胤恒\\Desktop\\‪11.png");
	imshow("source", source);
	cvtColor(source, grey, CV_BGR2GRAY);
	imshow("grey", grey);
	waitKey(30);
	return 0;
}