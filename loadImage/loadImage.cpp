#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>

using namespace cv;
using namespace std;

Mat convertToGrayScale(Mat mat);

/*  This function converst a RGB Mat object into a grayscale Mat.  */
Mat convertToGrayScale(Mat mat)
{
    int channels = mat.channels();

    int nRows = mat.rows;
    int nCols = mat.cols;

    Mat gmat = Mat(nRows, nCols, CV_8UC1);
    
    int i, j;
    for (i = 0; i < nRows; i++) 
    {
	for (j = 0; j < nCols; j++) 
	{
	    Vec3b pixel = mat.at<Vec3b>(i,j);
	    // Use formula of luminance
	    gmat.at<uchar>(i,j) = 0.3 * pixel[0] + 0.59 * pixel[1] + 0.11 * pixel[2]; 
	}
    }

    return gmat;
}

int main(int argc, char** argv) 
{
    String imageName("Image1.jpg");  // file path
    if (argc > 1) 
    {
	imageName = argv[1];
    }

    Mat image = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);    // read file

    if (image.empty()) 
    {
	cout << "Could not open or find the image" << std::endl;
	return -1;
    }

    Mat simage;
    cv::resize(image, simage, Size(), 0.05, 0.05);
//    imwrite("small2.jpg", simage);

    namedWindow("Display window", WINDOW_AUTOSIZE); // create window for display
    imshow("Display window", simage);	// show image inside window
    waitKey(0);

    cout << simage << endl;

    Mat drawline = Mat(300, 300, CV_8UC1, 255.0);
    line(drawline, Point(0, 20), Point(100, 100), Scalar(0), 2, 8);
    imshow("Display window", drawline);	// show image inside window
    waitKey(0);

    return 0;
}
