#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>

#include<iostream>
#include<string>
#include<cmath>

using namespace cv;
using namespace std;

Mat convertToGrayScale(Mat mat);
Mat mySobel(Mat mat);

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

Mat mySobel(Mat mat) 
{
    int nRows = mat.rows;
    int nCols = mat.cols;
    
    vector<vector<int> > Gx(nRows, vector<int>(nCols, 0));
    vector<vector<int> > Gy(nRows, vector<int>(nCols, 0));
    
    //    int Gx[nRows][nCols]; 
//    int Gy[nRows][nCols]; 

    Mat gradient = Mat(nRows, nCols, CV_8UC1);
    
    for (int i = 1; i < nRows-1; i++) 
    {
	for (int j = 1; j < nCols-1; j++) {
	   Gx.at(i).at(j) = mat.at<uchar>(i-1,j+1)+2*mat.at<uchar>(i,j+1)+mat.at<uchar>(i+1,j+1)
	              -mat.at<uchar>(i-1,j-1)-2*mat.at<uchar>(i,j-1)-mat.at<uchar>(i+1,j-1);

	   Gy.at(i).at(j) = mat.at<uchar>(i-1,j-1)+2*mat.at<uchar>(i-1,j)+mat.at<uchar>(i-1,j+1)
	              -mat.at<uchar>(i+1,j-1)-2*mat.at<uchar>(i+1,j)-mat.at<uchar>(i+1,j+1);

	   gradient.at<uchar>(i,j) = sqrt(pow(Gx.at(i).at(j), 2) + pow(Gy.at(i).at(j), 2));
	}
    }

    return gradient;
}

int main(int argc, char** argv) 
{
    String imageName("images/house.jpeg");  // file path
    if (argc > 1) 
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

    Mat image = imread(imageName, IMREAD_COLOR);    // read file

    if (image.empty()) 
    {
	cout << "Could not open or find the image" << std::endl;
	return -1;
    }

    Mat gimage;
    gimage = convertToGrayScale(image);

    namedWindow("Display window", WINDOW_AUTOSIZE); // create window for display
//    imshow("Display window", gimage);	// show image inside window
//    waitKey(0);

    Mat gradient = mySobel(gimage);
    imshow("Display window", gradient);
    waitKey(0);

    return 0;
}
