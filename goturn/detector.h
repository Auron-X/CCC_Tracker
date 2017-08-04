#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


class Detector {
	public:
	bool detectTM(Mat curFrame, Mat target, Rect2f& bb);
	bool detectCL(Mat curFrame, Mat &mask, Rect2f& bb);
};
