/**
 *  Computer Vision
 *  Project 1
 *  Shang-Hung Tsai
 */
 
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>

#include<iostream>
#include<string>
#include<cmath>

using namespace cv;
using namespace std;

Mat convertToGrayScale(Mat mat);
Mat mySobel(Mat mat, int threshold);

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

/* This function applies Sobel's Operator on the input image,
 * and normalize magnitude values to lie within range [0, 255].
 * Finally it performs a thresholding. */
Mat mySobel(Mat mat, int threshold) 
{
    int nRows = mat.rows;
    int nCols = mat.cols;
    int minM = INT_MAX;
    int maxM = INT_MIN;
    vector<vector<int> > M (nRows, vector<int>(nCols, 0));
    
    Mat gradient = Mat(nRows, nCols, CV_8UC1);
    
    for (int i = 1; i < nRows-1; i++) 
    {
	for (int j = 1; j < nCols-1; j++) {
	    int Gx = mat.at<uchar>(i-1,j+1)+2*mat.at<uchar>(i,j+1)+mat.at<uchar>(i+1,j+1)
	                     -mat.at<uchar>(i-1,j-1)-2*mat.at<uchar>(i,j-1)-mat.at<uchar>(i+1,j-1);

	    int Gy = mat.at<uchar>(i-1,j-1)+2*mat.at<uchar>(i-1,j)+mat.at<uchar>(i-1,j+1)
	                     -mat.at<uchar>(i+1,j-1)-2*mat.at<uchar>(i+1,j)-mat.at<uchar>(i+1,j+1);

	    M.at(i).at(j) = sqrt(pow(Gx, 2) + pow(Gy, 2));
	    
	    minM = min(minM, M.at(i).at(j));
	    maxM = max(maxM, M.at(i).at(j));
	}
    }

    cout << minM << endl;
    cout << maxM << endl;
    
    // normalize magnitude values and perform thresholding
    for (int i = 1; i < nRows-1; i++) 
    {
	for (int j = 1; j < nCols-1; j++) 
	{
	    int m = M.at(i).at(j);
	    m = ((float) (m - minM) / (float) (maxM-minM)) * 255;
//	    cout << m << endl;
	    if (m > threshold) {
		gradient.at<uchar>(i,j) = m;
	    } else {
		gradient.at<uchar>(i,j) = 0;
	    }
	}
    }

    return gradient;
}

int main(int argc, char** argv) 
{
    String imageName("images/house.jpeg");  // default image path
    if (argc > 1)   // get image name 
    {
	String one("1");
	String two("2");
	String three("3");
	if (one.compare(argv[1]) == 0) {
	    imageName = "images/TestImage1c.jpg";
	} else if (two.compare(argv[1]) == 0) {
	    imageName = "images/TestImage2c.jpg";
	} else if (three.compare(argv[1]) == 0) {
	    imageName = "images/TestImage3.jpg";
	} else {
	    imageName = argv[1];	
	} 
    }

    // Initialize display window
    namedWindow("Display window", WINDOW_AUTOSIZE); 

    // Read image
    Mat image = imread(imageName, IMREAD_COLOR);  
    if (image.empty()) 
    {
	cout << "Could not open or find the image" << std::endl;
	return -1;
    }

    // Display original image
    imshow("Display window", image);
    waitKey(0);

    // Convert image to gray scale and display result
    Mat gimage = convertToGrayScale(image);
    imshow("Display window", gimage);
    waitKey(0);

    // Apply Sobel operator and display result
    Mat gradient = mySobel(gimage, 10);
    imshow("Display window", gradient);
    waitKey(0);

    return 0;
}
