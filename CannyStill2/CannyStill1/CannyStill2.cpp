// CannyStill2.cpp
//Basic program to visualize the content of a webcam stgrem using OpenCV

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include <sstream>
#include <string>
#include <iostream>

#include<conio.h>           // it may be necessary to change or remove this line if not using Windows
using namespace cv;
using namespace std;

/// Global Variables
Mat img; Mat templ; Mat result;
char* image_window = "Source Image";
char* result_window = "Result window";

int match_method;
int max_Trackbar = 5;

/// Function Headers
void MatchingMethod(int, void*);


/**
* @function MatchingMethod
* @brief Trackbar callback
*/
void MatchingMethod(int, void*)
{
	/// Source image to display
	Mat img_display;
	img.copyTo(img_display);

	/// Create the result matrix
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;

	result.create(result_rows, result_cols, CV_32FC1);

	/// Do the Matching and Normalize
	matchTemplate(img, templ, result, match_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	/// Show me what you got
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0,0,255), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0,0,255), 2, 8, 0);

	imshow(image_window, img_display);
	imshow(result_window, result);

	return;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {
   /*cv::VideoCapture capWebcam(0);		// declare a VideoCapture object and associate to webcam, 0 => use 1st webcam

	if (capWebcam.isOpened() == false) {				// check if VideoCapture object was associated to webcam successfully
		std::cout << "error: capWebcam not accessed successfully\n\n";	// if not, print error message to std out
		return(0);														// and exit program
	}
	*/
	cv::VideoCapture capVideo;

	cv::Mat imgFrame;

	capVideo.open("768x576.avi");

	if (!capVideo.isOpened()) {                                                 // if unable to open video file
		std::cout << "\nerror reading video file" << std::endl << std::endl;      // show error message
		_getch();                    // it may be necessary to change or remove this line if not using Windows
		return(0);                                                              // and exit program
	}

	if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 1) {
		std::cout << "\nerror: video file must have at least one frame";
		_getch();
		return(0);
	}

	capVideo.read(imgFrame);

	char chCheckForEscKey = 0;

	while (capVideo.isOpened() && chCheckForEscKey != 27) {

		cv::imshow("imgFrame", imgFrame);

		// now we prepare for the next iteration

		if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {       // if there is at least one more frame
			capVideo.read(imgFrame);                            // read it
		}
		else {                                                  // else
			std::cout << "end of video\n";                      // show end of video message
			break;                                              // and jump out of while loop
		}

		chCheckForEscKey = cv::waitKey(1);      // get key press in case user pressed esc

	}

	if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
		cv::waitKey(0);                         // hold the windows open to allow the "end of video" message to show
	}
	// note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows

    Mat imgOriginal = imread("weber.jpg", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_COLOR); //read the image data in the file "MyPic.JPG" and store it in 'img'

	Mat imgColor(500, 1000, CV_8UC3, Scalar(0, 0, 100)); //create an image ( 3 channels, 8 bit image depth, 500 high, 1000 wide, (0, 0, 100) assigned for Blue, Green and Red plane respectively. )

	Mat imgZoom(imgOriginal, Rect(958, 1100, 160, 160)); // using a rectangle centered at the two first arguments

	if (imgOriginal.empty()) //check whether the image is loaded or not
	{
		cout << "Error : Image cannot be loaded..!!" << endl;
		system("pause"); //wait for a key press
		return -1;
	}

	if (imgZoom.empty()) //check whether the image is loaded or not
	{
		cout << "Error : Zoom image is empty..!!" << endl;
		system("pause"); //wait for a key press
		return -1;
	}

	Mat imgHSV;
	Mat imgThreshLow;
	Mat imgThreshHigh;
	Mat imgThresh;

	cv::cvtColor(imgOriginal, imgHSV, CV_BGR2HSV);

    // Apparently the HSV image is not normalized and the values of the matrix they still go from 0 to 255 for each entry
	// for example 0, 255, 255  is RED and in HSV percentage will be (0, 100, 100)

	cv::inRange(imgHSV, cv::Scalar(0, 0, 204), cv::Scalar(25, 25, 255), imgThreshLow);
	cv::inRange(imgHSV, cv::Scalar(0, 0, 255), cv::Scalar(10, 10, 255), imgThreshHigh);

	cv::add(imgThreshLow, imgThreshHigh, imgThresh);

	cv::GaussianBlur(imgThresh, imgThresh, cv::Size(3, 3), 0);

	cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

	cv::dilate(imgThresh, imgThresh, structuringElement);
	cv::erode(imgThresh, imgThresh, structuringElement);

	//Select the Mercedes symbol has template for the recognition 
	Mat imgTempl(imgThresh, Rect(958, 1100, 160, 160)); // using a rectangle centered at the two first arguments


	namedWindow("MainWindow", CV_WINDOW_NORMAL); //create a window with the name "MyWindow"
	imshow("MainWindow", imgOriginal); //display the image which is stored in the 'img' in the "MyWindow" window

												 // Create a window
	namedWindow("My Window", 1);

	//Create trackbar to change brightness
	int iSliderValue1 = 50;
	createTrackbar("Brightness", "My Window", &iSliderValue1, 100);

	//Create trackbar to change contrast
	int iSliderValue2 = 50;
	createTrackbar("Contrast", "My Window", &iSliderValue2, 100);

	while (true)
	{
		//Change the brightness and contrast of the image 
		Mat dst;
		int iBrightness = iSliderValue1 - 50;
		double dContrast = iSliderValue2 / 50.0;
		imgOriginal.convertTo(dst, -1, dContrast, iBrightness);

		//show the brightness and contrast adjusted image
		imshow("My Window", dst);

		// Wait until user press some key for 50ms
		int iKey = waitKey(50);

		//if user press 'ESC' key
		if (iKey == 27)
		{
			break;
		}
	}
	
	//namedWindow("ZoomWindow", CV_WINDOW_KEEPRATIO); //create a window with the name "MyWindow"
	//imshow("ZoomWindow", imgZoom); //display the image which is stored in the 'img' in the "MyWindow" window


	//erode and display the eroded image
	Mat imgErode, imgDil, imgInv;
    int element_shape = MORPH_RECT;
	int an = 2;
	Mat element = getStructuringElement(element_shape, Size(an * 2 + 1, an * 2 + 1), Point(an, an));
	erode(imgOriginal, imgErode, element);
	namedWindow("Template", CV_WINDOW_NORMAL);
	
	dilate(imgZoom, imgTempl, element);
	imshow("Template", imgTempl);

	//namedWindow("Dilate", CV_WINDOW_NORMAL);
	//imshow("Dilate", imgDil);
	/// Create windows
	img = imgOriginal;
	templ = imgTempl;

	namedWindow(image_window, CV_WINDOW_NORMAL);
	namedWindow(result_window, CV_WINDOW_NORMAL);

	/// Create Trackbar
	char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
	createTrackbar(trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod);
	MatchingMethod(0, 0);

	//Harris Corner Detection on the template 
	Mat grayTempl, harrTempl, Mc;

	vector<Vec3f> corners;

	cvtColor(imgTempl, grayTempl, COLOR_BGR2GRAY);

	cornerHarris(grayTempl, harrTempl, 2, 3, 0.04);
	dilate(harrTempl, harrTempl, element);

	
	namedWindow("HarrisTemplate", CV_WINDOW_NORMAL);

	imshow("HarrisTemplate", harrTempl);

	waitKey(0); //wait infinite time for a keypress

	destroyWindow("HarrisTemplate");
	destroyWindow("MainWindow"); //destroy the window with the name, "MainWindow"
	//destroyWindow("ZoomWindow"); //destroy the window with the name, "ZoomWindow"
	destroyWindow("Template");
	destroyWindow("my Window");
	//destroyWindow("Dilate");
	//destroyWindow("Inverte");

	/*char charCheckForEscKey = 0;

    while (charCheckForEscKey != 27 && capWebcam.isOpened()) {		// until the Esc key is pressed or webcam connection is lost
		bool blnFrameReadSuccessfully = capWebcam.read(imgOriginal);		// get next frame

		if (!blnFrameReadSuccessfully || imgOriginal.empty()) {		// if frame not read successfully
			std::cout << "error: frame not read from webcam\n";		// print error message to std out
			break;													// and jump out of while loop
		}

		// declare windows
		cv::namedWindow("imgOriginal", CV_WINDOW_AUTOSIZE);	// note: you can use CV_WINDOW_NORMAL which allows resizing the window

		cv::imshow("imgOriginal", imgOriginal);			// show windows

		charCheckForEscKey = cv::waitKey(1);			// delay (in ms) and get key press, if any
	}	// end while
   */

	return(0);
}