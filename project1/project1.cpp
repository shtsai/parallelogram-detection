/**
 *  Computer Vision
 *  Project 1
 *  Shang-Hung Tsai
 */
 
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc/imgproc.hpp>
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
void addLine(Mat mat, int p, float theta); 
void detectParallelogram(unordered_map<int, vector<int> > lines, Mat edges);
bool checkIntersection(int theta1, int a1p1, int a1p2, int theta2, int a2p1, int a2p2, Mat mat); 
bool findIntersection (int theta1, int p1, int theta2, int p2, Mat mat, vector<pair<int,int>> intersections, Mat pointMat);


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
 *
 * This function returns an unorder map, where key is theta and value is a
 * list of p values. 
 */
unordered_map<int, vector<int> > myHoughTransform(Mat mat) 
{
    bool display = true;
    int angleStep = 4;
    int pStep = 3;
    int maxP = sqrt(pow(mat.rows,2) + pow(mat.cols, 2));

    vector< vector<int> > M (maxP*2/pStep+1, vector<int> (180/angleStep+1, 0));
    vector< vector<pair<int,int> > > points ((maxP*2/pStep+1) * (180/angleStep+1), vector<pair<int, int> > (0));


    for (int i = 0; i < mat.rows; i++) 
    {
	for (int j = 0; j < mat.cols; j++) 
	{
	    if (mat.at<uchar>(i, j) == 0) // edge pixel
	    {
		// angle is the mean value of each bin
		for (float angle = angleStep/2; angle < 180; angle += angleStep) 
		{
		    float p = i * sin(angle * PI / 180) + j * cos(angle * PI / 180);
		    // cout << p << "+" << angle << " ";
		    // increment the corresponding accumulator
		    M.at((p + maxP)/pStep).at((int) angle/angleStep) += 1;
		    // add this point to the list for that line
		    points.at(((p + maxP)/pStep)*(180/angleStep+1)+(int) angle/angleStep).push_back(make_pair(i,j));
		}
	    }
	}
    }

    unordered_map<int, vector<int> > mp;

    for (int i = 1; i < M.size() - 1; i++) 
    {
	for (int j = 1; j < M.at(i).size() - 1; j++) 
	{
	    if (M.at(i).at(j) >= 250)  // threshold for straight lines
	    {
		bool localMaximum = true;
		for (int a = -4; a <= 4; a++) 
		{
		    for (int b = -4; b <= 4; b++)
		    {
			// check if this line is a local maximum
			if (i+a >= 0 && i+a < M.size() && j+b >= 0 && j+b < M.at(i).size()
			    && M.at(i+a).at(j+b) > M.at(i).at(j))
			{
			    localMaximum = false;
			    break;
			}
		    }
		    if (!localMaximum) break;
		}

		if (localMaximum) 
		{
		    cout << "For " << j * angleStep << ", " << i * pStep - maxP << endl;
		    vector<pair<int, int> > x = points.at(i * M.at(i).size() + j);
		    int disconnect = 0;
		    for (int ii  = 0; ii < x.size()-1; ii++) {
			if (abs(x[ii].first-x[ii+1].first) > 3 || abs(x[ii].second-x[ii+1].second > 3)) {
			    disconnect++;	
			}
//			cout << x[ii].first << " " << x[ii].second << endl;
		    }
//		    cout << (float) disconnect/x.size() << endl;

		    // only include lines that are dense enough
		    if ((float) disconnect/x.size() > 0.3) {
			if (mp.find(j * angleStep) == mp.end()) {
			    vector<int> v;
			    mp[j * angleStep] = v;
			}
		
			mp[j * angleStep].push_back(i * pStep - maxP);
		    }
		}
	    }
	}
    }

    /* add line to a mat for display purpose */
    if (display) {
//	cout << "Dimension " << mat.rows << " * " << mat.cols << endl;
	Mat lineMat = Mat(mat.rows, mat.cols, CV_8UC1, 0.0);
	for (auto it : mp)
	{
	    vector<int> list = it.second;
	    if (list.size() < 2)  // must have at least two lines 
	    {
		continue;
	    }
	    for (int i = 0; i < list.size(); i++) 
	    {
		addLine(lineMat, list.at(i), it.first + angleStep/2);
	    }
	}

	namedWindow("Display window 2", WINDOW_AUTOSIZE); 
        moveWindow("Display window 2", 20, 20);
        imshow("Display window 2", lineMat);
	waitKey(0);
    }

    return mp;
}

/* 
 * Given p and theta value, this function adds a line that satisfies the condition
 * to the input mat.
 */
void addLine(Mat mat, int p, float theta) 
{
    double COS = cos(theta * PI / 180);
    double TAN = tan(theta * PI / 180);

    for (int i = 0; i < mat.rows; i++) {
	int j = (p / COS) - TAN * i;
	if (j >= 0 && j < mat.cols) {
	    mat.at<uchar>(i, j) = 255;
	}
    }
}

void detectParallelogram(unordered_map<int, vector<int> > lines, Mat edges) 
{   

    // put all angles in an vector
    vector<int> angles;
    for (unordered_map<int, vector<int> >::iterator it = lines.begin(); it != lines.end(); it++)
    {
	/*
	cout << it->first << " ";
	for (int i = 0; i < it->second.size(); i++) {
	    cout << it->second.at(i) << " ";
	}
	cout << endl;
	*/
	angles.push_back(it->first);
    }

    for (int a1 = 0; a1 < angles.size(); a1++) {  // first angle
	for (int a2 = a1 + 1; a2 < angles.size(); a2++) { // second angle 
	    for (int a1p1 = 0; a1p1 < lines[angles.at(a1)].size(); a1p1++) {
		for (int a1p2 = a1p1 + 1; a1p2 < lines[angles.at(a1)].size(); a1p2++) {
		    for (int a2p1 = 0; a2p1 < lines[angles.at(a2)].size(); a2p1++) {
			for (int a2p2 = a2p1+1; a2p2 < lines[angles.at(a2)].size(); a2p2++) {
			    int theta1 = angles.at(a1);
			    int theta2 = angles.at(a2);
			    int a1p1v = lines[theta1].at(a1p1);
			    int a1p2v = lines[theta1].at(a1p2);
			    int a2p1v = lines[theta2].at(a2p1);
			    int a2p2v = lines[theta2].at(a2p2);

			    if (checkIntersection(theta1, a1p1v, a1p2v, theta2, a2p1v, a2p2v, edges)) {
				cout << "Find parallelogram" << endl;
			    }
			}
		    }
		}
	    }
	}
    }
}

bool checkIntersection(int theta1, int a1p1, int a1p2, int theta2, int a2p1, int a2p2, Mat mat) 
{
    Mat pointMat = Mat(mat.rows, mat.cols, CV_8UC1, 255.0);
    
    vector<pair<int,int> > intersections;

    if (!findIntersection(theta1, a1p1, theta2, a2p1, mat, intersections, pointMat)) {
	return false;
    }
    if (!findIntersection(theta1, a1p1, theta2, a2p2, mat, intersections, pointMat)) {
	return false;
    }
    if (!findIntersection(theta1, a1p2, theta2, a2p1, mat, intersections, pointMat)) {
	return false;
    }
    if (!findIntersection(theta1, a1p2, theta2, a2p2, mat, intersections, pointMat)) {
	return false;
    }

    if (true) {
	namedWindow("Display window 2", WINDOW_AUTOSIZE); 
        moveWindow("Display window 2", 20, 20);
        imshow("Display window 2", pointMat);
	waitKey(0);
    }

   /* 
    if (intersections.size() > 0) {
    for (int i = 0; i < intersections.size(); i++) {
	pair<int, int> p = intersections.at(i);
	cout << p.first << " + " << p.second << endl;
    }
    cout << endl;
    }
    */
    
    return true;
}

bool findIntersection (int theta1, int p1, int theta2, int p2, Mat mat, vector<pair<int,int>> intersections, Mat pointMat)
{
    double i, j;
    int inti, intj;

    i = (cos(theta1*PI/180)*p2 - cos(theta2*PI/180)*p1) 
	/ (sin(theta2*PI/180)*cos(theta1*PI/180)-sin(theta1*PI/180)*cos(theta2*PI/180));
    j = (p1 - i * sin(theta1*PI/180)) / cos(theta1*PI/180);

    inti = (int) i;
    intj = (int) j;
    bool found = false;

    // add points to the mat for display purpose
    cout << inti << " + " << intj << endl;
    for (int a = -2; a <= 2; a++) {
	for (int b = -2; b <= 2; b++) {
	    pointMat.at<uchar>(inti+a, intj+b) = 0;
	}
    }


    for (int a = -10; a <= 10; a++)  // check 5 * 5 surrounding pixels
    {
	for (int b = -10; b <= 10; b++) 
	{
	    if (found) 
	    {
		break;
	    }
	    if (inti+a >= 0 && inti+a < mat.rows && intj+b >= 0 && intj+b < mat.cols) 
	    {
		//cout << inti+a << "," << intj+b << "=" << (int) mat.at<uchar>(inti+a, intj+b) << " ";
		if (mat.at<uchar>(inti+a,intj+b)==0) 
		{
		    found = true;
		}
	    }
	}
    }
//    cout << endl;

    if (found) {
	pair<int, int> point (inti, intj);
	intersections.push_back(point);
	return true;
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
    }
    // Normalize intensity value to enhance contrast
    Mat enhanced = myEnhancer(gimage);
    if (DISPLAY_IMAGE) {
	imshow("Display window", enhanced);
	waitKey(0);
    }
    Mat gaussian = Mat(enhanced.rows, enhanced.cols, CV_8UC1, 0.0);
    GaussianBlur(enhanced, gaussian, Size(9, 9), 0, 0);
    if (DISPLAY_IMAGE) {
        imshow("Display window", gaussian);
	waitKey(0);
    }
    // Apply Sobel operator and display result
    Mat gradient = mySobel(enhanced);
    if (DISPLAY_IMAGE) {
        imshow("Display window", gradient);
	waitKey(0);
    }
    // Thresholding
    Mat edges = myThresholding(gradient, 11);
    if (DISPLAY_IMAGE) {
        imshow("Display window", gradient);
	waitKey(0);
    }
    // Hough Transform
    //unordered_map<int, vector<int> > lines = myHoughTransform(edges); 

    // Detect Parallelogram
    //detectParallelogram(lines, edges);
    

    return 0;
}
