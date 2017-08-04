#include <opencv2/opencv.hpp>
#include <iostream>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "./goturn/helper/bounding_box.h"
#include "./goturn/helper/image_proc.h"
#include "./goturn/goturn.h"
#include "./goturn/detector.h"

#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

using namespace std;
using namespace cv;
using namespace caffe;

static Mat image;
static bool paused;
static bool selectObjects = false;
static bool startSelection = false;
Rect2f boundingBox;


//Assign left/right cameras IDs 
int rightID = 1;
int leftID = 3;
bool active = 1;
int serverSocket = 0;

static void onMouse(int event, int x, int y, int, void*)
{
	if (!selectObjects)
	{
		switch (event)
		{
		case EVENT_LBUTTONDOWN:
			//set origin of the bounding box
			startSelection = true;
			boundingBox.x = x;
			boundingBox.y = y;
			boundingBox.width = boundingBox.height = 0;
			break;
		case EVENT_LBUTTONUP:
			//sei with and height of the bounding box
			boundingBox.width = std::abs(x - boundingBox.x);
			boundingBox.height = std::abs(y - boundingBox.y);
			paused = false;
			selectObjects = true;
			startSelection = false;
			break;
		case EVENT_MOUSEMOVE:

			if (startSelection && !selectObjects)
			{
				//draw the bounding box
				Mat currentFrame;
				image.copyTo(currentFrame);
				Mat hsv;
				cvtColor(currentFrame, hsv, CV_RGB2HSV);
				cout << (int)hsv.data[(y*hsv.cols + x) * 3 + 0] << " " << (int)hsv.data[(y*hsv.cols + x) * 3 + 1] << " " << (int)hsv.data[(y*hsv.cols + x) * 3 + 2] << endl;
				rectangle(currentFrame, Point((int)boundingBox.x, (int)boundingBox.y), Point(x, y), Scalar(255, 0, 0), 2, 1);
				imshow("GOTURN Tracking", currentFrame);
			}
			break;
		}
	}
}

void *checkSignal(void *data) {
	int n = 0;
	char recvBuff[1024];
	memset(recvBuff, '0', sizeof(recvBuff));
        
        while(1)
	{           
	   n = read(serverSocket, recvBuff, sizeof(recvBuff) - 1);
	   if (n > 0)
  		{
		active = true;
		cout << "Active signal!!!" << endl;
		}		
	}
}

void sendSignal(int id) {
	char sendBuff[10];
	sendBuff[0] = (char)(48 + id);	
	write(serverSocket, sendBuff, 1);
}

void connect2Server() {
	int n = 0;
	struct sockaddr_in serv_addr;

	if ((serverSocket = socket(AF_INET, SOCK_STREAM, 0)) < 0)
	{
		printf("\n Error : Could not create socket \n");

	}

	memset(&serv_addr, '0', sizeof(serv_addr));

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(5000);

	if (inet_pton(AF_INET, "172.17.0.1", &serv_addr.sin_addr) <= 0)
	{
		printf("\n inet_pton error occured\n");

	}

	while (connect(serverSocket, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
	{
		printf("\n Error : Connect Failed. Reconnecting... \n");
		sleep(1);
	}
        int flags = fcntl(serverSocket, F_GETFL, 0);
        fcntl(serverSocket, F_SETFL, flags | O_NONBLOCK);
	printf("Connected!\n");
}

void displayActive(Mat &img) {
string text = "Active Tracking";
int fontFace = FONT_HERSHEY_PLAIN;
double fontScale = 1.3;
int thickness = 2;  
cv::Point textOrg(50, 32);
cv::putText(img, text, textOrg, fontFace, fontScale, Scalar(0, 170, 0), thickness,12);
circle(img, Point(25,25), 15, Scalar(0, 200, 0), -1);
}

int main()
{
	const int camNum = 1;
	const int detectionFrequency = 5;
	bool showMask = 1;

	//Connect to Handover Server
	connect2Server();

	//Start listen thread
        pthread_t listen_thread;
        int listen_thread_id;
        if (listen_thread_id = pthread_create(&listen_thread, NULL, checkSignal, NULL))
    	{
    		printf("Error starting listening thread!!!");
    		return 0;
    	}
        cout << "Listening thread started..." << endl; 

	String proto_model = "tracker.prototxt";
	String caffe_model = "tracker.caffemodel";

	//Create GOTURN Tracker and detector
	GOTURN_Tracker tracker[4];
	for (int i = 0; i< camNum; i++)
		tracker[i].setup(proto_model, caffe_model);

	Detector detector;

	namedWindow("GOTURN Tracking", 1);
	setMouseCallback("GOTURN Tracking", onMouse, 0);

	//dataset->load("D:/ALOV300++");


	VideoCapture cap[camNum];

	for (int i = 0; i < camNum; i++) {
		cap[i].open(i);
		if (cap[i].isOpened() == false) {
			cout << "Camera #" << i << " error!!!";
			waitKey();
			return -1;
		}
	}

	//VideoCapture cap("http://192.168.2.2:81/videostream.cgi?user=admin&pwd=1004&resolution=640&rate=25");
	//VideoCapture cap("rtsp://admin:1004@192.168.2.2:10554/tcp/av0_0");
	//VideoCapture cap("http://192.168.2.2:81/livestream.cgi?user=admin&pwd=1004&streamid=2");
	//VideoCapture cap("http://api4.eye4.cn:808/login/eilab1004/starcraft/0/2/X/59d48ac57b71b3dd59ecbe5e25b7e85a6a6fa4d4/20170417154451/724/0/0/VSTC/0");

	//cap0.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	//cap0.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	//cap0.set(CV_CAP_PROP_FPS, 30);

	VideoWriter outputVideo;

	bool initialized = false;
	//paused = true;

	double timeStamp[100];

	Mat curFrame[4], displayFrame[4], mask[4];
	Rect2f trackedBB;
	Rect2f detectedBB;

	Mat targets;
	bool visible[4] = { false };	
	int cnt = 0;
	int notDetectedCounter = 0;
	for (int frameNum = 0; frameNum > -1; frameNum++)
		if (!paused)
		{
			for (int i = 0; i < camNum; i++) {
				cap[i] >> curFrame[i];
				displayFrame[i] = curFrame[i].clone();
			}

			int frameWidth, frameHeight;
			frameWidth = curFrame[0].cols;
			frameHeight = curFrame[0].rows;	
			

			//cout << curFrame[0].cols << "x" << curFrame[0].rows << endl;
			curFrame[0].copyTo(image);

			if (!initialized && selectObjects)
			{
				//Initialize tracker
				//tracker[0].Init(curFrame[0], boundingBox);
				initialized = true;
				timeStamp[50] = getTickCount();

				//targets = curFrame[0](boundingBox).clone();
				//visible[0] = true;
				//visible[1] = false;

				
				//Start video recording
				Size S = Size(frameWidth, frameHeight);
    				int codec = CV_FOURCC('M', 'J', 'P', 'G');    				
    				outputVideo.open("video.avi", codec, 10.0, S, true);
			
			}
			else if (initialized)

			{
				cnt++;				
				for (int k = 0; k < camNum; k++) {
					//Tracking
					if (visible[k] == true) {
						trackedBB = tracker[k].track(curFrame[k]);
						if (trackedBB.x < 3 ||
							trackedBB.y < 3 ||
							trackedBB.x + trackedBB.width > frameWidth - 3 ||
							trackedBB.y + trackedBB.height > frameHeight - 3)
						{
							
							visible[k] = false;							
							cout << "Target left a camera #" << k << " view!!!" << endl;
						}
						//cout << trackedBB.x << " " << trackedBB.y << " " << trackedBB.x + trackedBB.width << " " << trackedBB.y + trackedBB.height << endl;
						if (visible[k] == true) {							
							if (active)
								rectangle(displayFrame[k], trackedBB, Scalar(255, 0, 0), 2);
							//else
								//rectangle(displayFrame[k], trackedBB, Scalar(0, 255, 0), 2);
							targets = curFrame[k](trackedBB).clone();
						}
					}

					if (visible[k] == false || frameNum%detectionFrequency == 0) {
						//Detection every X frames
						if (detector.detectCL(curFrame[k], mask[k], detectedBB)) {
							tracker[k].Init(curFrame[k], detectedBB);
							visible[k] = true;
							if (frameNum%detectionFrequency != 0)
								cout << "Target appeared in camera #" << k << endl;
						}
						else
							visible[k] = false;
					}
					if (visible[k] == false)
						notDetectedCounter++;
					else
						notDetectedCounter = 0;
					if (notDetectedCounter > 10 && active)
					{
						cout << "XXXSDASD" << endl;
						sendSignal(rightID);
						active = false;			
					}
					if (active)
						displayActive(displayFrame[0]);
				}

			}

			//cout << "Frame #: " << frameNum << endl;
			imshow("GOTURN Tracking", displayFrame[0]);
			if (camNum > 1)
				imshow("GOTURN Tracking1", displayFrame[1]);
			if (camNum > 2)
				imshow("GOTURN Tracking2", displayFrame[2]);
			if (camNum > 3)
				imshow("GOTURN Tracking3", displayFrame[3]);




			//outputVideo.write(displayFrame[0]);
			char c = waitKey(1);
			if (c == 'q')
				break;
		}
	timeStamp[51] = getTickCount();
	double seconds = (timeStamp[51] - timeStamp[50]) / getTickFrequency();
	cout << "Totap FPS: " << cnt / seconds << endl;
	cout << "Press any button to exit";

	waitKey(0);
	return 1;
}
