// ImageSubtractionCpp.sln
// main.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<conio.h>           // it may be necessary to change or remove this line if not using Windows

#include "Blob.h"

// global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {

	cv::VideoCapture capWebcam(0);		// declare a VideoCapture object and associate to webcam, 0 => use 1st webcam
	
	cv::Mat imgFrame1;
	cv::Mat imgFrame2;

	if (capWebcam.isOpened() == false) {				// check if VideoCapture object was associated to webcam successfully
		std::cout << "error: capWebcam not accessed successfully\n\n";	// if not, print error message to std out
		return(0);														// and exit program
	}

	char chCheckForEscKey = 0;

	while (capWebcam.isOpened() && chCheckForEscKey != 27) {

		std::vector<Blob> blobs;

		cv::Mat imgDifference;
		cv::Mat imgThresh;

		capWebcam.read(imgFrame1);
		bool blnFrame1ReadSuccessfully = capWebcam.read(imgFrame1);		// get next frame

		if (!blnFrame1ReadSuccessfully || imgFrame1.empty()) {		// if frame not read successfully
			std::cout << "error: frame not read from webcam\n";		// print error message to std out
			break;													// and jump out of while loop
		}
		cv::Mat imgFrame1Copy = imgFrame1.clone();

		capWebcam.read(imgFrame2);
		bool blnFrame2ReadSuccessfully = capWebcam.read(imgFrame2);		// get next frame

		if (!blnFrame2ReadSuccessfully || imgFrame1.empty()) {		// if frame not read successfully
			std::cout << "error: frame not read from webcam\n";		// print error message to std out
			break;													// and jump out of while loop
		}
		cv::Mat imgFrame2Copy = imgFrame2.clone();

		cv::cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
		cv::cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);

		cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
		cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

		cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

		cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);

		cv::imshow("imgThresh", imgThresh);

		cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
		cv::Mat structuringElement9x9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

		cv::dilate(imgThresh, imgThresh, structuringElement5x5);
		cv::dilate(imgThresh, imgThresh, structuringElement5x5);
		cv::erode(imgThresh, imgThresh, structuringElement5x5);

		cv::Mat imgThreshCopy = imgThresh.clone();

		std::vector<std::vector<cv::Point> > contours;

		cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		cv::Mat imgContours(imgThresh.size(), CV_8UC3, SCALAR_BLACK);

		cv::drawContours(imgContours, contours, -1, SCALAR_WHITE, -1);

		cv::imshow("imgContours", imgContours);

		std::vector<std::vector<cv::Point> > convexHulls(contours.size());

		for (unsigned int i = 0; i < contours.size(); i++) {
			cv::convexHull(contours[i], convexHulls[i]);
		}

		for (auto &convexHull : convexHulls) {
			Blob possibleBlob(convexHull);

			if (possibleBlob.boundingRect.area() > 100 &&
				possibleBlob.dblAspectRatio >= 0.2 &&
				possibleBlob.dblAspectRatio <= 1.2 &&
				possibleBlob.boundingRect.width > 15 &&
				possibleBlob.boundingRect.height > 20 &&
				possibleBlob.dblDiagonalSize > 30.0) {
				blobs.push_back(possibleBlob);
			}
		}

		cv::Mat imgConvexHulls(imgThresh.size(), CV_8UC3, SCALAR_BLACK);

		convexHulls.clear();

		for (auto &blob : blobs) {
			convexHulls.push_back(blob.contour);
		}

		cv::drawContours(imgConvexHulls, convexHulls, -1, SCALAR_WHITE, -1);

		cv::imshow("imgConvexHulls", imgConvexHulls);

		imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

		for (auto &blob : blobs) {                                                  // for each blob
			cv::rectangle(imgFrame2Copy, blob.boundingRect, SCALAR_RED, 2);             // draw a red box around the blob
			cv::circle(imgFrame2Copy, blob.centerPosition, 3, SCALAR_GREEN, -1);        // draw a filled-in green circle at the center
		}

		cv::imshow("imgFrame2Copy", imgFrame2Copy);

		// now we prepare for the next iteration

		imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is

		chCheckForEscKey = cv::waitKey(1);      // get key press in case user pressed esc

	}
	
	if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
		cv::waitKey(0);                         // hold the windows open to allow the "end of video" message to show
	}
	// note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows

	return(0);
}






















