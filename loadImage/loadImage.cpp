#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>

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
    String imageName("../images/house.jpeg");  // file path
    if (argc > 1) 
    {
	imageName = argv[1];
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
    imshow("Display window", gimage);	// show image inside window

    waitKey(0);
    return 0;
}
