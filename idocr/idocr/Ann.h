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
Mat getR(const Mat Input);
double distance(Point a, Point b);
void OstuBeresenThreshold(const Mat &in, Mat &out);
void normalPosArea(const Mat &intputImg, RotatedRect &rects_optimal, Mat& output_area);
void char_segment(const Mat &inputImg, vector<Mat> &dst_mat);
void getAnnXML();
void calcGradientFeat(const Mat &imgSrc, Mat &out);
float sumMatValue(const Mat &image);
Mat projectHistogram(const Mat &img, int t);
void ann_train(Ptr<ANN_MLP> &ann, int numCharacters, int nlayers);
void classify(Ptr<ANN_MLP> &ann, vector<Mat> &char_Mat, vector<int> &char_result);
void getParityBit(vector<int> &char_result);
void findarea(const Mat &Input, RotatedRect &rect, Point estimate, int MODE);
bool is_SandABcorect(const RotatedRect &candidate, int MODE);
void watch_mat(Mat Input,string s);