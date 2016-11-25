#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
//#include <opencv2/ml/ml.hpp> 
#include <opencv/ml.h>
#include "Ann.h"
#define NUMBER 1
#define NAME 2
using namespace std;
using namespace cv;
using namespace ml;
Mat ConverToRGray(const Mat Input)
{
	Mat splitBGR[3];
	split(Input, splitBGR);
	return splitBGR[2];
}
double Distance(Point a, Point b)
{
	return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}
void EstimatePosition(Mat Input, Point &numesti, Point &namesti, Point &minzuesti)
{
	const string face_cascade_name = "haarcascade_frontalface_alt.xml";
	String nestedCascadeName = "./haarcascade_eye.xml";
	CascadeClassifier face_cascade, nestedCascade;
	if (!face_cascade.load(face_cascade_name)) {
		printf("级联分类器错误，可未找到文件，拷贝该文件到工程目录下！\n");
		exit(-1);
	}
	if (!nestedCascade.load(nestedCascadeName))
	{
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
		exit(-1);
	}
	vector<Rect> faces;
	Mat face_gray;
	face_cascade.detectMultiScale(Input, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(1, 1));
	Point center(faces[0].x + faces[0].width*0.5, faces[0].y + faces[0].height*0.5);

		cout << "x=" << faces[0].x << "y=" << faces[0].y << endl;
		cout << "x=" << center.x << "y=" << center.y << endl;
		cout << "width=" << faces[0].width << "height=" << faces[0].height << endl;
		ellipse(Input, center, Size(faces[0].width*0.5, faces[0].height*0.5), 0, 0, 360, Scalar(255, 0, 0), 2, 7, 0);

	float x = faces[0].width, y = faces[0].height;
	Rect rect(center.x - 2.26*x+ 3.07*x/2, center.y + 1.1*y+ 0.4255*y/2,10,10);
	numesti = Point(center.x - 2.26*x + 3.07*x / 2, center.y + 1.1*y + 0.4255*y / 2);
	namesti = Point(center.x - 2.4*x, center.y - 0.8*y);
	minzuesti= Point(center.x - 1.75*x, center.y - 0.45*y);
	ellipse(Input, namesti, Size(10, 10), 0, 0, 360, Scalar(255, 0, 0), 2, 7, 0);
	ellipse(Input, minzuesti, Size(10, 10), 0, 0, 360, Scalar(255, 0, 0), 2, 7, 0);
	imshow("身份证号码", Input);
}
void OstuThreshold(const Mat &in, Mat &out) //输入为单通道
{

	double ostu_T = threshold(in, out, 0, 255, CV_THRESH_OTSU); //otsu获得全局阈值

	double min;
	double max;
	minMaxIdx(in, &min, &max);
	const double CI = 0.12;
	double beta = CI*(max - min + 1) / 128;
	double beta_lowT = (1 - beta)*ostu_T;
	double beta_highT = (1 + beta)*ostu_T;

	Mat doubleMatIn;
	in.copyTo(doubleMatIn);
	int rows = doubleMatIn.rows;
	int cols = doubleMatIn.cols;
	double Tbn;
	for (int i = 0; i < rows; ++i)
	{
		//获取第 i行首像素指针
		uchar * p = doubleMatIn.ptr<uchar>(i);
		uchar *outPtr = out.ptr<uchar>(i);

		//对第i 行的每个像素(byte)操作
		for (int j = 0; j < cols; ++j)
		{

			if (i <2 | i>rows - 3 | j<2 | j>rows - 3)
			{

				if (p[j] <= beta_lowT)
					outPtr[j] = 0;
				else
					outPtr[j] = 255;
			}
			else
			{
				Tbn = sum(doubleMatIn(Rect(i - 2, j - 2, 5, 5)))[0] / 25;  //窗口大小25*25
				if (p[j] < beta_lowT | (p[j] < Tbn && (beta_lowT <= p[j] && p[j] >= beta_highT)))
					outPtr[j] = 0;
				if (p[j] > beta_highT | (p[j] >= Tbn && (beta_lowT <= p[j] && p[j] >= beta_highT)))
					outPtr[j] = 255;
			}
		}
	}

}
void Normalize(const Mat &intputImg, RotatedRect &rects_optimal, Mat& output_area)
{
	float r, angle;

	angle = rects_optimal.angle;
	r = (float)rects_optimal.size.width / (float)(float)rects_optimal.size.height;
	if (r<1)
		angle = 90 + angle;
	Mat rotmat = getRotationMatrix2D(rects_optimal.center, angle, 1);//获得变形矩阵对象
	Mat img_rotated;
	warpAffine(intputImg, img_rotated, rotmat, intputImg.size(), CV_INTER_CUBIC);

	//裁剪图像
	Size rect_size = rects_optimal.size;

	if (r<1)
		std::swap(rect_size.width, rect_size.height);
	Mat  img_crop;
	getRectSubPix(img_rotated, rect_size, rects_optimal.center, img_crop);

	//用光照直方图调整所有裁剪得到的图像，使具有相同宽度和高度，适用于训练和分类
	Mat resultResized;
	resultResized.create(20, 300, CV_8UC1);
	cv::resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);
	resultResized.copyTo(output_area);
}
void CharSegment(const Mat &inputImg, vector<Mat> &dst_mat)
{
	imshow(",inpputImg", inputImg);
	waitKey(0);
	Mat img_threshold;

	Mat whiteImg(inputImg.size(), inputImg.type(), cv::Scalar(255));
	Mat in_Inv = whiteImg - inputImg;

	// threshold(in_Inv ,img_threshold , 140,255 ,CV_THRESH_BINARY ); //反转黑白色
	threshold(in_Inv, img_threshold, 0, 255, CV_THRESH_OTSU); //大津法二值化

	int x_char[19] = { 0 };
	short counter = 1;
	short num = 0;
	bool flag[50000];

	for (int j = 0; j < img_threshold.cols; ++j)
	{
		flag[j] = true;
		for (int i = 0; i < img_threshold.rows; ++i)
		{

			if (img_threshold.at<uchar>(i, j) != 0)
			{
				flag[j] = false;
				break;
			}

		}
	}

	for (int i = 0; i < img_threshold.cols - 2; ++i)
	{
		if (flag[i] == true)
		{
			x_char[counter] += i;
			num++;
			if (flag[i + 1] == false && flag[i + 2] == false)
			{
				x_char[counter] = x_char[counter] / num;
				num = 0;
				counter++;
			}
		}

	}
	imshow("1", in_Inv);
	waitKey(0);
	x_char[18] = img_threshold.cols;
	imshow("1", Mat(in_Inv, Rect(x_char[2], 0, x_char[2 + 1] - x_char[2], img_threshold.rows)));
	for (int i = 0; i < 18; i++)
	{
		dst_mat.push_back(Mat(in_Inv, Rect(x_char[i], 0, x_char[i + 1] - x_char[i], img_threshold.rows)));
	}


	// imwrite("b.jpg" , img_threshold);

}
void GetAnnXML() // 此函数仅需运行一次，目的是获得训练矩阵和标签矩阵，保存于ann_xml.xml中
{
	FileStorage fs("ann_xml.xml", FileStorage::WRITE);
	Mat  trainData;
	Mat classes = Mat::zeros(1, 550, CV_8UC1);   //1700*48维，也即每个样本有48个特征值
	char path[60];
	Mat img_read;
	for (int i = 0; i<=10; i++)  //第i类
	{
		for (int j = 1; j<= 50; ++j)  //i类中第j个，即总共有11类字符，每类有50个样本
		{
			sprintf(path, "C:\\Number_char\\%d\\%d (%d).png", i, i, j);
			img_read = imread(path, 0);
			Mat dst_feature;
			GradientFeat(img_read, dst_feature); //计算每个样本的特征矢量
			trainData.push_back(dst_feature);

			classes.at<uchar>(i * 50 + j - 1) = i;
		}
	}
	fs << "TrainingData" << trainData;
	fs << "classes" << classes;
	fs.release();
}
void GradientFeat(const Mat &Input, Mat &Output)
{
	vector <float>  feat;
	Mat image;

	resize(Input, image, Size(8, 16));

	// 计算x方向和y方向上的滤波
	float mask[3][3] = { { 1, 2, 1 },{ 0, 0, 0 },{ -1, -2, -1 } };

	Mat y_mask = Mat(3, 3, CV_32F, mask) / 8;
	Mat x_mask = y_mask.t(); // 转置
	Mat sobelX, sobelY;

	filter2D(image, sobelX, CV_32F, x_mask);
	filter2D(image, sobelY, CV_32F, y_mask);

	sobelX = abs(sobelX);
	sobelY = abs(sobelY);

	float totleValueX = SumMatValue(sobelX);
	float totleValueY = SumMatValue(sobelY);

	// 将图像划分为4*2共8个格子，计算每个格子里灰度值总和的百分比
	for (int i = 0; i < image.rows; i = i + 4)
	{
		for (int j = 0; j < image.cols; j = j + 4)
		{
			Mat subImageX = sobelX(Rect(j, i, 4, 4));
			feat.push_back(SumMatValue(subImageX) / totleValueX);
			Mat subImageY = sobelY(Rect(j, i, 4, 4));
			feat.push_back(SumMatValue(subImageY) / totleValueY);
		}
	}

	//计算第2个特征
	Mat imageGray;
	//cvtColor(imgSrc,imageGray,CV_BGR2GRAY);
	cv::resize(Input, imageGray, Size(4, 8));
	Mat p = imageGray.reshape(1, 1);
	p.convertTo(p, CV_32FC1);
	for (int i = 0; i<p.cols; i++)
	{
		feat.push_back(p.at<float>(i));
	}

	//增加水平直方图和垂直直方图
	Mat vhist = ProjectHistogram(Input, 1); //水平直方图
	Mat hhist = ProjectHistogram(Input, 0);  //垂直直方图
	for (int i = 0; i<vhist.cols; i++)
	{
		feat.push_back(vhist.at<float>(i));
	}
	for (int i = 0; i<hhist.cols; i++)
	{
		feat.push_back(hhist.at<float>(i));
	}


	Output = Mat::zeros(1, feat.size(), CV_32F);
	for (int i = 0; i<feat.size(); i++)
	{
		Output.at<float>(i) = feat[i];
	}
}
float SumMatValue(const Mat &image)
{
	float sumValue = 0;
	int r = image.rows;
	int c = image.cols;
	if (image.isContinuous())
	{
		c = r*c;
		r = 1;
	}
	for (int i = 0; i < r; i++)
	{
		const uchar* linePtr = image.ptr<uchar>(i);
		for (int j = 0; j < c; j++)
		{
			sumValue += linePtr[j];
		}
	}
	return sumValue;
}
Mat ProjectHistogram(const Mat &img, int t)
{
	Mat lowData;
	cv::resize(img, lowData, Size(8, 16)); //缩放到8*16

	int sz = (t) ? lowData.rows : lowData.cols;
	Mat mhist = Mat::zeros(1, sz, CV_32F);

	for (int j = 0; j < sz; j++)
	{
		Mat data = (t) ? lowData.row(j) : lowData.col(j);
		mhist.at<float>(j) = countNonZero(data);
	}

	double min, max;
	minMaxLoc(mhist, &min, &max);

	if (max > 0)
		mhist.convertTo(mhist, -1, 1.0f / max, 0);

	return mhist;
}
void Train(Ptr<ANN_MLP> &ann, int numCharacters, int nlayers)
{
	Mat trainData, classes;
	FileStorage fs;
	fs.open("ann_xml.xml", FileStorage::READ);

	fs["TrainingData"] >> trainData;
	fs["classes"] >> classes;
	Mat layerSizes(1, 3, CV_32SC1);     //3层神经网络
	layerSizes.at<int>(0,0) = trainData.cols;   //输入层的神经元结点数，设置为24
	layerSizes.at<int>(0,1) = nlayers; //1个隐藏层的神经元结点数，设置为24
	layerSizes.at<int>(0,2) = numCharacters; //输出层的神经元结点数为:10
	//cout << layerSizes.at<int>(0, 0)<<layerSizes.at<int>(0, 0) << layerSizes.at<int>(0, 1) << layerSizes.at<int>(0, 2);
	ann->setLayerSizes(layerSizes);
	ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
	ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5000, 0.01));
	ann->setTrainMethod(ANN_MLP::BACKPROP, 0.001);
	

	Mat trainClasses;
	trainClasses.create(trainData.rows, numCharacters, CV_32FC1);
	for (int i = 0; i< trainData.rows; i++)
	{
		for (int k = 0; k< trainClasses.cols; k++)
		{
			if (k == (int)classes.at<uchar>(i))
			{
				trainClasses.at<float>(i, k) = 1;
			}
			else
				trainClasses.at<float>(i, k) = 0;

		}

	}
	Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainClasses);
	ann->train(tData);
}
void classify(Ptr<ANN_MLP> &ann, vector<Mat> &char_Mat, vector<int> &char_result)
{
	char_result.resize(char_Mat.size());
	for (int i = 0; i<char_Mat.size(); ++i)
	{
		Mat output(1, 10, CV_32FC1); //1*10矩阵

		Mat char_feature;
		GradientFeat(char_Mat[i], char_feature);
		ann->predict(char_feature, output);
		Point maxLoc;
		double maxVal;
		minMaxLoc(output, 0, &maxVal, 0, &maxLoc);

		char_result[i] = maxLoc.x;

	}
}
void LastBit(vector<int> &char_result)
{
	int mod = 0;
	int wights[17] = { 7,9,10,5,8,4 ,2,1,6,3,7,9,10,5,8,4,2 };
	for (int i = 0; i < 17; ++i)
		mod += char_result[i] * wights[i];//乘相应系数求和

	mod = mod % 11; //对11求余

	int value[11] = { 1,0,10,9,8,7,6,5,4,3,2 };
	char_result[17] = value[mod];
}
void FindArea(const Mat &Input, RotatedRect &rect,Point estimate,int MODE)
{


	Mat threshold_R;
	OstuThreshold(Input, threshold_R); //二值化


	Mat imgInv(Input.size(), Input.type(), cv::Scalar(255));
	Mat threshold_Inv = imgInv - threshold_R; //黑白色反转，即背景为黑色

	Mat element = getStructuringElement(MORPH_RECT, Size(15, 3));  //闭形态学的结构元素
	morphologyEx(threshold_Inv, threshold_Inv, CV_MOP_CLOSE, element);

	vector< vector <Point> > contours;
	findContours(threshold_Inv, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//只检测外轮廓
				
																					//对候选的轮廓进行进一步筛选
	////查看
	//Mat watch_Input;
	//Input.copyTo(watch_Input);
	//for (int i = 0; i < contours.size(); i++)
	//{
	//	//使用边界框的方式  
	//	Rect aRect = boundingRect(contours[i]);
	//	if (aRect.area() <= 100) continue;
	//	int tmparea = aRect.height*aRect.height;
	//	//if (((double)aRect.width / (double)aRect.height>5) && tmparea >= 200 && tmparea <= 2500)
	//	{
	//		rectangle(watch_Input, cvPoint(aRect.x, aRect.y), cvPoint(aRect.x + aRect.width, aRect.y + aRect.height), 1, 2);
	//		//cvDrawContours( dst, contours, color, color, -1, 1, 8 );  
	//	}
	//}
	//imshow("watch_Input", watch_Input);
	//waitKey(30);
	double mindis = 100000;
	int maxi = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		RotatedRect mr = minAreaRect(Mat(contours[i]));
		if (!IsSuit(mr,MODE))  //判断矩形轮廓是否符合要求
		{
			continue;
		}
		else
		{
			Point center =mr.center;
			if (mindis > Distance(center, estimate))
			{
				maxi = i;
				mindis = Distance(center, estimate);
			}
		}
	}
	rect= minAreaRect(Mat(contours[maxi]));
	//测试是否找到了号码区域
	//    Mat out;
	//    in.copyTo(out);
	//    Point2f vertices[4];
	//    rects[0].points(vertices);
	//    for (int i = 0; i < 4; i++)
	//    line(out, vertices[i], vertices[(i+1)%4], Scalar(0,0,0));//画黑色线条
	//    imshow("Test_Rplane" ,out);
	//    waitKey(0);


}
bool IsSuit(const RotatedRect &candidate,int MODE)
{
	switch (MODE)
	{
	case NUMBER:
	{
		float error = 0.2;
		const float aspect = 4.5 / 0.3; //长宽比
		int min = 10 * aspect * 10; //最小区域
		int max = 50 * aspect * 50;  //最大区域
		float rmin = aspect - aspect*error; //考虑误差后的最小长宽比
		float rmax = aspect + aspect*error; //考虑误差后的最大长宽比

		int area = candidate.size.height * candidate.size.width;
		double r = (double)candidate.size.width / (double)candidate.size.height;
		if (r < 1)
			r = 1 / r;

		if ((area < min || area > max) || (r< rmin || r > rmax)) //满足该条件才认为该candidate为车牌区域
			return false;
		else
			return true; 
		break;
	}
	case NAME:
		break;
	default:
		return false;
		break;
	}
	
	
}

void WatchMat(Mat Input,string s)
{
	imshow(s, Input);
	waitKey(30);
}


vector<int> GetNumber(Mat Input)//gray input
{
	vector<Mat> char_mat;
	CharSegment(Input, char_mat);
	//getAnnXML();
	Ptr<ANN_MLP> ann;
	ann = ANN_MLP::create();
	Train(ann, 10, 24); //10为输出层结点,也等于输出的类别，24为隐藏层结点

	vector<int> char_result;
	classify(ann, char_mat, char_result);

	LastBit(char_result); //最后一位易出错，直接由前17位计算最后一位

	return char_result;
}

string GetName(Mat Input)
{
	string ans;
	imwrite("name.jpg", Input);
	system("Tesseract-OCR\\tesseract.exe name.jpg name -l chi_sim");
	freopen("name.txt", "r", stdin);
	cin >> ans;
	return ans;
}
void adjust(Mat Inut, Mat &Output)
{

}
void test(int n)
{
	string path = "C:\\test\\";
	for (int i = 1; i <= n; i++)
	{
		Mat source = imread(path + to_string(i) + ".jpg");
		resize(source, source, Size(400, 300));
		Mat imgRplane = ConverToRGray(source); //获得原始图像R分量
		Mat testr;
		imgRplane.copyTo(testr);
		Point numesti, namesti, minzuesti;
		EstimatePosition(source, numesti, namesti, minzuesti);
		RotatedRect  rect;
		FindArea(imgRplane, rect, numesti, NUMBER);  //获得身份证号码区域
		Mat outputMat;
		Normalize(imgRplane, rect, outputMat); //获得身份证号码字符矩阵
		vector<int> num = GetNumber(outputMat);
		cout << endl;
		for (int i = 0; i <= 17; i++)cout << num[i];
	}
}

int main()
{

	//GetAnnXML();
	test(1);
	return 0;
}