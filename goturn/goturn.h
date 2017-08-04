#include <iostream>
#include <string>

#include "./helper/bounding_box.h"
#include "./helper/image_proc.h"

#include <caffe/caffe.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace caffe;

class GOTURN_Tracker {
	public:
		void setup(std::string proto_model, std::string caffe_model);		
		
		void Init(Mat frame,  Rect2f& boundingBox);
		Rect2f track(Mat& curFrame);
		
		void GetFeatures(const string& feature_name, std::vector<float>* output);
		void GetOutput(std::vector<float>* output);
		void WrapInputLayer(std::vector<cv::Mat>* target_channels, std::vector<cv::Mat>* image_channels);
		void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
		void Estimate(const cv::Mat& image, const cv::Mat& target, std::vector<float>* output);
		void Regress(const cv::Mat& image_curr, const cv::Mat& image, const cv::Mat& target, BoundingBox* bbox);

		int num_channels_;
		Size input_geometry_;
		Mat mean_;
		boost::shared_ptr<Net<float> > net_;
		
		Mat prevFrame;
		BoundingBox bbox_prev_tight_;
		BoundingBox	bbox_curr_prior_tight_;
};

