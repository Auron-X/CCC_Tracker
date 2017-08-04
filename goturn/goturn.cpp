#include "goturn.h"

using namespace std;

void GOTURN_Tracker::setup(string proto_model, string caffe_model) {
	Caffe::set_mode(caffe::Caffe::GPU);
	net_.reset(new Net<float>(proto_model, caffe::TEST));
	net_->CopyTrainedLayersFrom(caffe_model);

	num_channels_ = 3;
	input_geometry_ = Size(227, 227);
	mean_ = Mat(input_geometry_, CV_32FC3, cv::Scalar(104, 117, 123));
}

void GOTURN_Tracker::Init(Mat frame, Rect2f& boundingBox) {
	prevFrame = frame;

	bbox_prev_tight_.x1_ = boundingBox.x;
	bbox_prev_tight_.y1_ = boundingBox.y;
	bbox_prev_tight_.x2_ = boundingBox.x + boundingBox.width;
	bbox_prev_tight_.y2_ = boundingBox.y + boundingBox.height;

	bbox_curr_prior_tight_.x1_ = boundingBox.x;
	bbox_curr_prior_tight_.y1_ = boundingBox.y;
	bbox_curr_prior_tight_.x2_ = boundingBox.x + boundingBox.width;
	bbox_curr_prior_tight_.y2_ = boundingBox.y + boundingBox.height;
}

Rect2f GOTURN_Tracker::track(Mat& curFrame){
	// Get target from previous image.
	cv::Mat target_pad;

	CropPadImage(bbox_prev_tight_, prevFrame, &target_pad);

	// Crop the current image based on predicted prior location of target.
	cv::Mat curr_search_region;
	BoundingBox search_location;
	double edge_spacing_x, edge_spacing_y;
	CropPadImage(bbox_curr_prior_tight_, curFrame, &curr_search_region, &search_location, &edge_spacing_x, &edge_spacing_y);

	// Estimate the bounding box location of the target, centered and scaled relative to the cropped image.
	BoundingBox bbox_estimate;
	Regress(curFrame, curr_search_region, target_pad, &bbox_estimate);


	//regressor->Regress(image_curr, curr_search_region, target_pad, &bbox_estimate);
	// Unscale the estimation to the real image size.
	BoundingBox bbox_estimate_unscaled;
	bbox_estimate.Unscale(curr_search_region, &bbox_estimate_unscaled);

	// Find the estimated bounding box location relative to the current crop.
	BoundingBox* bbox_estimate_uncentered = new BoundingBox;
	bbox_estimate_unscaled.Uncenter(curFrame, search_location, edge_spacing_x, edge_spacing_y, bbox_estimate_uncentered);

	// Save the image.
	prevFrame = curFrame.clone();

	// Save the current estimate as the location of the target.
	bbox_prev_tight_ = *bbox_estimate_uncentered;

	// Save the current estimate as the prior prediction for the next image.
	// TODO - replace with a motion model prediction?
	bbox_curr_prior_tight_ = *bbox_estimate_uncentered;

	Rect2f result;
	result.x = bbox_curr_prior_tight_.x1_;
	result.y = bbox_curr_prior_tight_.y1_;
	result.width = bbox_curr_prior_tight_.x2_ - bbox_curr_prior_tight_.x1_;
	result.height = bbox_curr_prior_tight_.y2_ - bbox_curr_prior_tight_.y1_;
	
	return result;
}

void GOTURN_Tracker::GetFeatures(const string& feature_name, std::vector<float>* output) {
	//printf("Getting %s features\n", feature_name.c_str());

	// Get a pointer to the requested layer.
	const boost::shared_ptr<Blob<float> > layer = net_->blob_by_name(feature_name.c_str());

	// Compute the number of elements in this layer.
	int num_elements = 1;
	for (int i = 0; i < layer->num_axes(); ++i) {
		const int elements_in_dim = layer->shape(i);
		//printf("Layer %d: %d\n", i, elements_in_dim);
		num_elements *= elements_in_dim;
	}
	//printf("Total num elements: %d\n", num_elements);

	// Copy all elements in this layer to a vector.
	const float* begin = layer->cpu_data();
	const float* end = begin + num_elements;
	*output = std::vector<float>(begin, end);
}
void GOTURN_Tracker::GetOutput(std::vector<float>* output) {
	// Get the fc8 output features of the network (this contains the estimated bounding box).
	GetFeatures("fc8", output);
}
void GOTURN_Tracker::WrapInputLayer(std::vector<cv::Mat>* target_channels, std::vector<cv::Mat>* image_channels) {
	Blob<float>* input_layer_target = net_->input_blobs()[0];
	Blob<float>* input_layer_image = net_->input_blobs()[1];

	int target_width = input_layer_target->width();
	int target_height = input_layer_target->height();
	float* target_data = input_layer_target->mutable_cpu_data();
	for (int i = 0; i < input_layer_target->channels(); ++i) {
		cv::Mat channel(target_height, target_width, CV_32FC1, target_data);
		target_channels->push_back(channel);
		target_data += target_width * target_height;
	}

	int image_width = input_layer_image->width();
	int image_height = input_layer_image->height();
	float* image_data = input_layer_image->mutable_cpu_data();
	for (int i = 0; i < input_layer_image->channels(); ++i) {
		cv::Mat channel(image_height, image_width, CV_32FC1, image_data);
		image_channels->push_back(channel);
		image_data += image_width * image_height;
	}
}
void GOTURN_Tracker::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
	// Convert the input image to the input image format of the network.
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, CV_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, CV_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, CV_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, CV_GRAY2BGR);
	else
		sample = img;

	// Convert the input image to the expected size.
	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	// Convert the input image to the expected number of channels.
	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	// Subtract the image mean to try to make the input 0-mean.
	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	// This operation will write the separate BGR planes directly to the
	// input layer of the network because it is wrapped by the cv::Mat
	// objects in input_channels.
	cv::split(sample_normalized, *input_channels);

	/*CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
	== net_->input_blobs()[0]->cpu_data())
	<< "Input channels are not wrapping the input layer of the network.";*/
}
void GOTURN_Tracker::Estimate(const cv::Mat& image, const cv::Mat& target, std::vector<float>* output) {
	assert(net_->phase() == caffe::TEST);

	// Reshape the input blobs to be the appropriate size.
	Blob<float>* input_target = net_->input_blobs()[0];
	input_target->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);

	Blob<float>* input_image = net_->input_blobs()[1];
	input_image->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);

	Blob<float>* input_bbox = net_->input_blobs()[2];
	input_bbox->Reshape(1, 4, 1, 1);

	// Forward dimension change to all layers.
	net_->Reshape();

	// Process the inputs so we can set them.
	std::vector<cv::Mat> target_channels;
	std::vector<cv::Mat> image_channels;
	WrapInputLayer(&target_channels, &image_channels);

	// Set the inputs to the network.
	Preprocess(image, &image_channels);
	Preprocess(target, &target_channels);

	// Perform a forward-pass in the network.
	net_->Forward();

	// Get the network output.
	GetOutput(output);
}
void GOTURN_Tracker::Regress(const cv::Mat& image_curr, const cv::Mat& image, const cv::Mat& target, BoundingBox* bbox) {
	assert(net_->phase() == caffe::TEST);

	// Estimate the bounding box location of the target object in the current image.
	std::vector<float> estimation;
	Estimate(image, target, &estimation);

	// Wrap the estimation in a bounding box object.
	*bbox = BoundingBox(estimation);
}
