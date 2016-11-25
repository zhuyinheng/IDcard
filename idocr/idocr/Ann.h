#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
//#include <opencv2/ml/ml.hpp> 
#include <opencv/ml.h>
using namespace std;
using namespace cv;
using namespace ml;
Mat ConverToRGray(const Mat Input);
Mat ProjectHistogram(const Mat &img, int t);
double Distance(Point a, Point b);
void OstuThreshold(const Mat &in, Mat &out);
void Normalize(const Mat &intputImg, RotatedRect &rects_optimal, Mat& output_area);
void CharSegment(const Mat &inputImg, vector<Mat> &dst_mat);
void GetAnnXML();
void GradientFeat(const Mat &imgSrc, Mat &out);
float SumMatValue(const Mat &image);
void Train(Ptr<ANN_MLP> &ann, int numCharacters, int nlayers);
void classify(Ptr<ANN_MLP> &ann, vector<Mat> &char_Mat, vector<int> &char_result);
void LastBit(vector<int> &char_result);
void FindArea(const Mat &Input, RotatedRect &rect, Point estimate, int MODE);
bool IsSuit(const RotatedRect &candidate, int MODE);
void WatchMat(Mat Input,string s);
void test(int n);
