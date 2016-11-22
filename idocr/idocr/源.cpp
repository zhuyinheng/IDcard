#include <iostream>
#include<vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


Point numesti;
Mat source, gray;
Mat getR(const Mat Input)
{
	Mat splitBGR[3];
	split(Input, splitBGR);
	return splitBGR[2];
}
double distance(Point a, Point b)
{
	return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}
Mat numberarea(Mat Input)
{
	Mat Bin(Input.rows, Input.cols, CV_8UC1);;
	threshold(Input, Bin , 85, 255, THRESH_BINARY_INV);
	const Mat element = getStructuringElement(MORPH_RECT, Size(15, 3));
	morphologyEx(Bin, Bin, CV_MOP_CLOSE, element); 
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(Bin, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
	
	for (int i = 0; i < contours.size(); i++)
	{
		Rect Rect = boundingRect(contours[i]);
		rectangle(source, Point(Rect.x, Rect.y), Point(Rect.x + Rect.width, Rect.y + Rect.height), 1, 2);
	}
	if(contours.size()>0)
	{ 
		double mindis = 100000;
		int maxi=0;
		for (int i = 0; i < contours.size(); i++)
		{
			Rect Rect = boundingRect(contours[i]);
			Point center = Point(Rect.x+Rect.width/2, Rect.y+Rect.height/2);
			if (mindis > distance(center, numesti))
			{
				maxi = i;
				mindis = distance(center, numesti);
			}
		}
		cout << maxi << endl << mindis;
		return source(boundingRect(contours[maxi]));
	}
	
	return Bin;
}
void estipos(Mat Input)
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
	Rect rect(center.x - 2.26*x+ 3.07*x/2, center.y + 1.1*y+ 0.4255*y/2,50,50);
	numesti = Point(center.x - 2.26*x + 3.07*x / 2, center.y + 1.1*y + 0.4255*y / 2);
	imshow("身份证号码", Input(rect));
}

int main()
{


	source = imread("C:\\Users\\朱胤恒\\Desktop\\11.jpg");
	resize(source, source, Size(400, 300));
	
	gray = getR(source);
	estipos(source);
	//imshow("so", numberarea(gray));
	//imshow("source", source);









	waitKey(0);
	return 0;
}