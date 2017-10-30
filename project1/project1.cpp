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
#include<unordered_map>

#define PI 3.1415926
#define DISPLAY_IMAGE true 

using namespace cv;
using namespace std;

Mat convertToGrayScale(Mat mat);
Mat myEnhancer(Mat mat);
Mat mySobel(Mat mat);
Mat myThresholding(Mat mat, int threshold);
unordered_map<int, vector<int> > myHoughTransform(Mat mat);
void addLine(Mat mat, int p, int theta); 


/*  This function converst a RGB Mat object into a grayscale Mat.  */
Mat convertToGrayScale(Mat mat)
{
    int channels = mat.channels();

    int nRows = mat.rows;
    int nCols = mat.cols;

    Mat gmat = Mat(nRows, nCols, CV_8UC1, 0.0);
    
    int i, j;
    for (i = 0; i < nRows; i++) 
    {
	for (j = 0; j < nCols; j++) 
	{
	    Vec3b pixel = mat.at<Vec3b>(i,j);
	    // Use formula of luminance
	    gmat.at<uchar>(i,j) = 0.3 * pixel[2] + 0.59 * pixel[1] + 0.11 * pixel[0]; 
	}
    }

    return gmat;
}

/*
 * This function enhances the contrast of an image by spread out gray value 
 * distributions. Output image will use all 255 gray level values.
 */
Mat myEnhancer(Mat mat) 
{
    uchar maxp = 0;
    uchar minp = 255;
    
    for (int i = 0; i < mat.rows; i++) {  // find max and min pixel value
	for (int j = 0; j < mat.cols; j++) {
	    maxp = max(maxp, mat.at<uchar>(i, j));
	    minp = min(minp, mat.at<uchar>(i, j));
	}
    }

    for (int i = 0; i < mat.rows; i++) {
	for (int j = 0; j < mat.cols; j++) {
	    mat.at<uchar>(i, j) = ((float) mat.at<uchar>(i,j)-minp) / (float) (maxp-minp) * 255;
	}
    } 

    return mat;
}

/* 
 * This function applies Sobel's Operator on the input image,
 * and normalize magnitude values to lie within range [0, 255]. 
 * */
Mat mySobel(Mat mat) 
{
    int nRows = mat.rows;
    int nCols = mat.cols;
    int minM = INT_MAX;
    int maxM = INT_MIN;
    vector<vector<int> > M (nRows, vector<int>(nCols, 0));
    
    Mat gradient = Mat(nRows, nCols, CV_8UC1, 0.0);
    
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

    // normalize magnitude values 
    for (int i = 1; i < nRows-1; i++) 
    {
	for (int j = 1; j < nCols-1; j++) 
	{
	    gradient.at<uchar>(i,j) = ((float) (M.at(i).at(j) - minM) / (float) (maxM-minM)) * 255;
	}
    }

    return gradient;
}

/* This function performs thresholding on the given mat */
Mat myThresholding(Mat mat, int threshold) 
{
    for (int i = 0; i < mat.rows; i++) 
    {
	for (int j = 0; j < mat.cols; j++) 
	{
	    if (mat.at<uchar>(i,j)  > threshold) 
	    {
		mat.at<uchar>(i,j) = 0;
	    } else {
		mat.at<uchar>(i,j) = 255;
	    }
	}
    }
    return mat;
}

/* 
 * My implementation of Hough Transform.
 * Here the angle is defined as the angle between perpendicular line 
 * and the positve x axis, counter-clockwise.
 * This is because we are using (i,j) coordinate system.
 */
unordered_map<int, vector<int> > myHoughTransform(Mat mat) 
{
    bool display = false;
    int angleStep = 2;
    int pStep = 3;
    int maxP = sqrt(pow(mat.rows,2) + pow(mat.cols, 2));

    vector< vector<int> > M (maxP*2/pStep+1, vector<int> (180/angleStep+1, 0));
    // cout << maxP*2/pStep << endl;
    // cout << mat << endl;

    for (int i = 0; i < mat.rows; i++) 
    {
	for (int j = 0; j < mat.cols; j++) 
	{
	    if (mat.at<uchar>(i, j) == 0) // edge pixel
	    {
		for (int angle = 0; angle < 180; angle += angleStep) 
		{
		    float p = i * sin(angle * PI / 180) + j * cos(angle * PI / 180);
		    // cout << p << "+" << angle << " ";
		    M.at((p + maxP)/pStep).at(angle/angleStep) += 1;
		}
	    }
	}
    }

    Mat lineMat = Mat(mat.rows, mat.cols, CV_8UC1, 0.0);
//    vector< vector<int> >lines;
    unordered_map<int, vector<int> > mp;

    for (int i = 0; i < M.size(); i++) 
    {
	for (int j = 0; j < M.at(i).size(); j++) 
	{
	    if (M.at(i).at(j) >= 300) {
		if (mp.find(j * angleStep) == mp.end()) {
		    vector<int> v;
		    mp[j * angleStep] = v;
		}
		
		mp[j * angleStep].push_back(i * pStep - maxP);

		/* add p and theta to a 2D vector
		vector<int> param (2, 0);
		param.at(0) = i * pStep - maxP;
		param.at(1) = j * angleStep;
		lines.push_back(param);
		*/

		/* add line to a mat for display purpose */
		if (display) {
		    addLine(lineMat, i * pStep - maxP, j * angleStep);
		}
	    }
	}
    }

    if (display) {
	namedWindow("Display window 2", WINDOW_AUTOSIZE); 
        moveWindow("Display window 2", 20, 20);
        imshow("Display window 2", lineMat);
    }

    return mp;
}

/* 
 * Given p and theta value, this function adds a line that satisfies the condition
 * to the input mat.
 */
void addLine(Mat mat, int p, int theta) {
    double COS = cos(theta * PI / 180);
    double TAN = tan(theta * PI / 180);

    for (int i = 0; i < mat.rows; i++) {
	int j = (p / COS) - TAN * i;
	if (j >= 0 && j < mat.cols) {
	    mat.at<uchar>(i, j) = 255;
	}
    }
}

int main(int argc, char** argv) 
{
    String imageName("images/house.jpeg");  // default image path
    if (argc > 1)   // get image name 
    {
	String one("1");
	String two("2");
	String three("3");
	String four("4");
	String five("5");
	if (one.compare(argv[1]) == 0) {
	    imageName = "images/TestImage1c.jpg";
	} else if (two.compare(argv[1]) == 0) {
	    imageName = "images/TestImage2c.jpg";
	} else if (three.compare(argv[1]) == 0) {
	    imageName = "images/TestImage3.jpg";
	} else if (four.compare(argv[1]) == 0) {
	    imageName = "images/small.jpg";
	} else if (five.compare(argv[1]) == 0) {
	    imageName = "images/small2.jpg";
	} else {
	    imageName = argv[1];	
	} 
    }

    // Initialize display window
    namedWindow("Display window", WINDOW_AUTOSIZE); 
    moveWindow("Display window", 20, 20);

    // Read image
    Mat image = imread(imageName, IMREAD_COLOR);  
    if (image.empty()) 
    {
	cout << "Could not open or find the image" << std::endl;
	return -1;
    }


    // Display original image
    if (DISPLAY_IMAGE) {
        imshow("Display window", image);
	waitKey(0);
    }

    // Convert image to gray scale and display result
    Mat gimage = convertToGrayScale(image);
    if (DISPLAY_IMAGE) {
        imshow("Display window", gimage);
	waitKey(0);
	Mat enhanced = myEnhancer(gimage);
	imshow("Display window", enhanced);
	waitKey(0);
    }

    // Apply Sobel operator and display result
    Mat gradient = mySobel(gimage);
    if (DISPLAY_IMAGE) {
        imshow("Display window", gradient);
	waitKey(0);
    }

    // Thresholding
    Mat edges = myThresholding(gradient, 15);
    if (DISPLAY_IMAGE) {
        imshow("Display window", gradient);
	waitKey(0);
    }

    // Hough Transform
    unordered_map<int, vector<int> > lines = myHoughTransform(edges); 

    return 0;
}
