#include "detector.h"

bool Detector::detectTM(Mat curFrame, Mat target,  Rect2f& bb) {
	int result_cols = curFrame.cols - target.cols + 1;
	int result_rows = curFrame.rows - target.rows + 1;
	Mat result(result_cols, result_rows, CV_32FC1);

	

	int match_method = CV_TM_CCOEFF_NORMED;
	matchTemplate(curFrame, target, result, match_method);

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	cout << "Max prob: " << maxVal << endl;
	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}
	if (maxVal > 0.6) {
		bb.x = matchLoc.x;
		bb.y = matchLoc.y;
		bb.width = target.cols;
		bb.height = target.rows;
		return true;
	}

	return false;
}

bool Detector::detectCL(Mat curFrame, Mat &mask, Rect2f& bb){
	Mat backgrd;
	Mat hsv;
	Mat lower_red_hue_range;
	Mat upper_red_hue_range;

	cvtColor(curFrame, hsv, CV_BGR2HSV);

	inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), lower_red_hue_range);
	inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255), upper_red_hue_range);
	addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, mask);

	//inRange(curFrame, Scalar(0, 0, 150), Scalar(110, 60, 255), mask);
	//inRange(curFrame, Scalar(0, 0, 0), Scalar(255, 255, 255), mask);
	imshow("TTT", mask);
	
	int morph_elem = 2;
	int morph_size = 5;
	int count = 0;
	Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	//morphologyEx(mask, mask, MORPH_OPEN, element);
	//dilate(mask, mask, element);
	
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	RNG rng;
	findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	/// Find the convex hull object for each contour
	vector<vector<Point> >hull(contours.size());
	
	for (int i = 0; i < contours.size(); i++)
	{
		convexHull(Mat(contours[i]), hull[i], false);
	}
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect;
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
                Rect tmp = boundingRect(Mat(contours_poly[i]));
		if (tmp.area() > 30) 
			boundRect.push_back(tmp);
	}
        //cout << "Blob num: " << contours.size() << "   Detected rect: " << boundRect.size() << endl;
	if (boundRect.size() == 0)
		return false;
	else {
		bb = boundRect[0];
		for (int i = 1; i < boundRect.size() - 1; i++) {
			float x1, x2, y1, y2;
			x1 = min(bb.x, (float)boundRect[i].x);
			y1 = min(bb.y, (float)boundRect[i].y);
			x2 = max(bb.x+bb.width, (float)(boundRect[i].x+boundRect[i].width));
			y2 = max(bb.y + bb.height, (float)(boundRect[i].y + boundRect[i].height));
			bb.x = x1;
			bb.y = y1;
			bb.width = max(x2 - x1, (float)10);
			bb.height = max(y2 - y1, (float)10);
		}
	}
	return true;
}
