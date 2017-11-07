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
#define ANGLESTEP 3.0
#define PSTEP 3.0

using namespace cv;
using namespace std;

Mat convertToGrayScale(Mat mat);
Mat myEnhancer(Mat mat);
Mat mySobel(Mat mat);
Mat myThresholding(Mat mat, int threshold);
unordered_map<double, vector<int> > myHoughTransform(Mat mat);
bool isPointsClose(vector<pair<int,int> >points, double closeness, double SD);
void addLine(Mat mat, int p, float theta); 
void addLineByPoints(Mat mat, vector<pair<int,int> >  points, int intensity); 
vector<vector<pair<int, int> > > detectParallelogram(unordered_map<double, vector<int> > lines, Mat edges);
vector<pair<int,int> > checkIntersection(double theta1, int a1p1, int a1p2, double theta2, int a2p1, int a2p2, Mat edges); 
bool findIntersection (double theta1, int p1, double theta2, int p2, vector<pair<int,int> > &intersection, Mat edges, Mat pointMat);
void displayParallelogram(vector<vector<pair<int, int> > > intersections, Mat image);

int maxP;
int cols;
int rows;
int ACCU_THRESHOLD;
double CLOSENESS;
int STANDARD_DEVIATION;
int GRADIENT_THRESHOLD;

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

unordered_map<double, vector<int> > myHoughTransform(Mat mat)
{
    int maxP = sqrt(pow(rows, 2) + pow(cols, 2));
    int psize = maxP * 2 / PSTEP;
    int anglesize = 180/ANGLESTEP;
    int* accumulator = new int[psize * anglesize];
    vector<pair<int,int> >* points = new vector<pair<int, int> >[psize * anglesize];
    unordered_map<double, vector<int> > lines;

    for (int i = 0; i < psize; i++) {
	for (int j = 0; j < anglesize; j++) {
	    accumulator[i * anglesize + j] = 0;
	}
    }

    for (int y = 0; y < mat.rows; y++) {
	for (int x = 0; x < mat.cols; x++) {
	    if (mat.at<uchar>(y, x) == 0) {  // edge point
		for (float theta = ANGLESTEP/2; theta < 180; theta += ANGLESTEP) {
		    float p = x * cos(theta * PI / 180) + y * sin(theta * PI / 180);
		    int pindex = (p + maxP) / PSTEP;
		    int tindex = theta/ANGLESTEP;
		    accumulator[pindex * anglesize + tindex]++;
		    points[pindex * anglesize + tindex].push_back(make_pair(y, x));
		}
	    }
	}
    }

    Mat lineMat = Mat(rows, cols, CV_8UC1, 0.0);
    for (int i = 0; i < psize; i++) {
	for (int j = 0; j < anglesize; j++) {
	    if (accumulator[i * anglesize + j] > ACCU_THRESHOLD) {
		bool localMaximum = true;
		for (int a = -4; a <= 4; a++) {
		    for (int b = -4; b <= 4; b++) {
			if (i+a >= 0 && i+a < psize && j+b >= 0 && j+b < anglesize && accumulator[i*anglesize+j] < accumulator[(i+a)*anglesize + (j+b)]) {
			    localMaximum = false;
			    break;
			}
		    }
		    if (!localMaximum) break;
		}

		if (localMaximum) {
		    if (!isPointsClose(points[i * anglesize + j], CLOSENESS, STANDARD_DEVIATION)) {
			continue;   // skip lines that don't meet the requirement
		    }

		    /*
		    if (lines.find(j * ANGLESTEP + ANGLESTEP/2.0) != lines.end()) {  // there are other lines with the same angle
			lines[j * ANGLESTEP + ANGLESTEP/2.0].push_back(i * PSTEP - maxP);
		    } else if (j - 1 >= 0 && lines.find((j-1) * ANGLESTEP + ANGLESTEP/2.0) != lines.end()) {
			lines[(j-1) * ANGLESTEP + ANGLESTEP/2.0].push_back(i * PSTEP - maxP);
		    } else if (j + 1 < anglesize && lines.find((j+1) * ANGLESTEP + ANGLESTEP/2.0) != lines.end()) {
			lines[(j+1) * ANGLESTEP + ANGLESTEP/2.0].push_back(i * PSTEP - maxP);
		    } else {
			lines[j * ANGLESTEP + ANGLESTEP/2.0].push_back(i * PSTEP - maxP);
		    } 
		    */

		    lines[j * ANGLESTEP + ANGLESTEP/2.0].push_back(i * PSTEP - maxP);
//		    cout << j * ANGLESTEP + ANGLESTEP/2.0 << " " << i * PSTEP - maxP << endl;
		    if (false) {
			addLine(lineMat, i * PSTEP - maxP, j * ANGLESTEP + ANGLESTEP/2);
		    }
//		    addLineByPoints(lineMat, points[i * anglesize + j], 255);
		    if (false) {    // add one line per mat
			lineMat.release();
//			lineMat = mat.clone();
			lineMat = Mat(rows, cols, CV_8UC1, 0.0);
			addLineByPoints(lineMat, points[i * anglesize + j], 255);
			namedWindow("Display window 2", WINDOW_AUTOSIZE); 
			moveWindow("Display window 2", 20, 20);
			imshow("Display window 2", lineMat);
			waitKey(0);
		    }
		}

	    }
	}
    }


    if (false) {
        namedWindow("Display window 2", WINDOW_AUTOSIZE); 
	moveWindow("Display window 2", 20, 20);
        imshow("Display window 2", lineMat);
	waitKey(0);
    }
    
    delete accumulator;
    return lines;
}

/* Given a list of points, test whether the points are close enough to each other. */
bool isPointsClose(vector<pair<int,int> >points, double closeness, double SD)
{
    double sd = 0;
    int sumx = 0;
    int sumy = 0;
    int discontinous = 0;
    for (int i = 0; i < points.size()-1; i++) {
	int x1 = points.at(i).first;
	int y1 = points.at(i).second;
	int x2 = points.at(i+1).first;
	int y2 = points.at(i+1).second;
	if (abs(x1-x2) > 10 || abs(y1-y2) > 10) {
	    discontinous++;
	}
	sumx += points.at(i).first;
	sumy += points.at(i).second;
    }
    double rate = (double) discontinous / points.size();

    sumx += points.at(points.size()-1).first;
    sumy += points.at(points.size()-1).second;
    double meanx = sumx / points.size();
    double meany = sumy / points.size();
    for (int i = 0; i < points.size(); i++) {
	sd += sqrt(pow(points.at(i).first-meanx, 2) + pow(points.at(i).second-meany, 2));
    }
    double meansd = sd / points.size();

    return rate < closeness && meansd < SD;
}

/*
 * This is function plots all the edge points corresponding to a line on a Mat.
 */
void addLineByPoints(Mat mat, vector<pair<int,int> > points, int intensity) {
    for (int i = 0; i < points.size(); i++) {
	int y = points.at(i).first;
	int x = points.at(i).second;
        mat.at<uchar>(y, x) = intensity;
    }
}

/* 
 * Given p and theta value, this function adds a line that satisfies the condition
 * to the input mat.
 */
void addLine(Mat mat, int p, float theta)
{
    double COS = cos(theta * PI / 180);
    double TAN = tan(theta * PI / 180);
    for (int y = 0; y < mat.rows; y++) {
	int x = (p / COS) - TAN * y;
	if (x >= 0 && x < mat.cols) {
	    mat.at<uchar>(y, x) = 255;
	}
    }
}

vector<vector<pair<int, int> > > detectParallelogram(unordered_map<double, vector<int> > lines, Mat edges) 
{   
    // put all angles in an vector
    vector<double> angles;
    for (unordered_map<double, vector<int> >::iterator it = lines.begin(); it != lines.end(); it++)
    {
	/*
	cout << it->first << ": ";
	for (int i = 0; i < it->second.size(); i++) {
	    cout << it->second.at(i) << " ";
	}
	cout << endl;
	*/
	if (it->second.size() > 1) {	// need at least two lines for each angle
	    angles.push_back(it->first);
	}
    }
    vector<vector<pair<int, int> > > intersections;
    for (int a1 = 0; a1 < angles.size(); a1++) {  // first angle
	for (int a2 = a1 + 1; a2 < angles.size(); a2++) { // second angle 
	    for (int a1p1 = 0; a1p1 < lines[angles.at(a1)].size(); a1p1++) {
		for (int a1p2 = a1p1 + 1; a1p2 < lines[angles.at(a1)].size(); a1p2++) {
		    for (int a2p1 = 0; a2p1 < lines[angles.at(a2)].size(); a2p1++) {
			for (int a2p2 = a2p1+1; a2p2 < lines[angles.at(a2)].size(); a2p2++) {
			    double theta1 = angles.at(a1);
			    double theta2 = angles.at(a2);
			    int a1p1v = lines[theta1].at(a1p1);
			    int a1p2v = lines[theta1].at(a1p2);
			    int a2p1v = lines[theta2].at(a2p1);
			    int a2p2v = lines[theta2].at(a2p2);

			    vector<pair<int,int> > intersect = checkIntersection(theta1, a1p1v, a1p2v, theta2, a2p1v, a2p2v, edges);
			    if (intersect.size() == 4) {
				cout << "Find parallelogram" << endl;
				intersections.push_back(intersect);
			    }
			}
		    }
		}
	    }
	}
    }
/*
    Mat lineMat = Mat(rows, cols, CV_8UC1, 255.0);
    for (int i = 0; i < intersections.size(); i++) {
	vector<pair<int, int>> parallel = intersections.at(i);
	for (int j = 0; j < parallel.size(); j++) {
	    pair<int, int> p = parallel.at(j);
	    pair<int, int> p2;
	    if (j == parallel.size() - 1) {
		p2 = parallel.at(0);
	    } else {
	        p2 = parallel.at(j+1);
	    }
	    line(lineMat, Point(p.second, p.first), Point(p2.second, p2.first), Scalar(0), 2, 8);
	}
    }
   	
    namedWindow("Display window 2", WINDOW_AUTOSIZE); 
    moveWindow("Display window 2", 20, 20);
    imshow("Display window 2", lineMat);
    waitKey(0);
*/
    return intersections;
}

vector<pair<int,int> > checkIntersection(double theta1, int a1p1, int a1p2, double theta2, int a2p1, int a2p2, Mat edges) 
{
    vector<pair<int, int> > intersection;
    Mat pointMat = Mat(rows, cols, CV_8UC1, 255.0);
    

    if (!findIntersection(theta1, a1p1, theta2, a2p1, intersection, edges, pointMat)) {
	intersection.clear();
    }
    if (!findIntersection(theta2, a2p1, theta1, a1p2, intersection, edges, pointMat)) {
	intersection.clear();
    }
    if (!findIntersection(theta1, a1p2, theta2, a2p2, intersection, edges, pointMat)) {
	intersection.clear();
    }
    if (!findIntersection(theta2, a2p2, theta1, a1p1, intersection, edges, pointMat)) {
	intersection.clear();
    }
/*
    if (intersection.size() == 4) {
	for (int j = 0; j < intersection.size(); j++) {
	    pair<int, int> p = intersection.at(j);
	    pair<int, int> p2;
	    if (j == intersection.size() - 1) {
		p2 = intersection.at(0);
	    } else {
	        p2 = intersection.at(j+1);
	    }
	    line(pointMat, Point(p.second, p.first), Point(p2.second, p2.first), Scalar(0), 2, 8);
	}
	namedWindow("Display window 2", WINDOW_AUTOSIZE); 
        moveWindow("Display window 2", 20, 20);
        imshow("Display window 2", pointMat);
	waitKey(0);
    }
*/
    /*
    if (intersection.size() > 0) {
    for (int i = 0; i < intersection.size(); i++) {
	pair<int, int> p = intersection.at(i);
	cout << p.first << " + " << p.second << endl;
    }
    cout << endl;
    }
    */
   
    return intersection;
}

bool findIntersection (double theta1, int p1, double theta2, int p2, vector<pair<int,int> > &intersection, Mat edges, Mat pointMat)
{
    double i, j;
    int inti, intj;

    i = (cos(theta1*PI/180)*p2 - cos(theta2*PI/180)*p1) 
	/ (sin(theta2*PI/180)*cos(theta1*PI/180)-sin(theta1*PI/180)*cos(theta2*PI/180));
    j = (p1 - i * sin(theta1*PI/180)) / cos(theta1*PI/180);

    if (i < 0 || i >= rows || j < 0 || j >= cols) return false;
    inti = (int) i;
    intj = (int) j;
    bool found = true;

    for (int di = -9; di <= 9; di++) {
	for (int dj = -9; dj <= 9; dj++) {
	    if (inti + di >= 0 && inti + di < rows && intj + dj >= 0 && intj + dj < cols) {
		if (edges.at<uchar>(inti+di, intj+dj) == 0) {
		    found = true;
		    break;
	        }
	    }
	}
	if (found) break;
    } 

    if (found) {
	/*
	// add points to the mat for display purpose
        cout << inti << " + " << intj << endl;
	for (int a = -2; a <= 2; a++) {
	    for (int b = -2; b <= 2; b++) {
		pointMat.at<uchar>(inti+a, intj+b) = 0;
	    }
        }
	*/

    /*
	cout << "found" << endl;
	cout << inti << " " << intj << endl;
    */

	pair<int, int> point (inti, intj);
	intersection.push_back(point);
	return true;
    }
    return false;
}
    
void displayParallelogram(vector<vector<pair<int, int> > > intersections, Mat image) {
    Mat lineMat = image.clone();
    for (int i = 0; i < intersections.size(); i++) {
	cout << "Coordinates of corners: {";
	vector<pair<int, int>> parallel = intersections.at(i);
	for (int j = 0; j < parallel.size(); j++) {
	    pair<int, int> p = parallel.at(j);
	    pair<int, int> p2;
	    if (j == parallel.size() - 1) {
		p2 = parallel.at(0);
	    } else {
	        p2 = parallel.at(j+1);
	    }
	    line(lineMat, Point(p.second, p.first), Point(p2.second, p2.first), Scalar(0, 0, 255), 2, 8);
	    cout << "(" << p.second << ",-" << p.first << ")";
	}
	cout << "}" << endl;
    }	    
    namedWindow("Display window 2", WINDOW_AUTOSIZE); 
    moveWindow("Display window 2", 20, 20);
    imshow("Display window 2", lineMat);
    waitKey(0);
}

int main(int argc, char** argv) 
{
    String imageName("images/house.jpeg");  // default image path
    if (argc > 1)   // get image name and step up configuration parameters 
    {
	String one("1");
	String two("2");
	String three("3");
	String four("4");
	String five("5");
	if (one.compare(argv[1]) == 0) {
	    imageName = "images/TestImage1c.jpg";
	    ACCU_THRESHOLD = 500;
	    CLOSENESS = 0.01;
	    STANDARD_DEVIATION = 90;
	    GRADIENT_THRESHOLD = 30;
	} else if (two.compare(argv[1]) == 0) {
	    imageName = "images/TestImage2c.jpg";
	    ACCU_THRESHOLD = 100;
	    CLOSENESS = 0.045;
	    STANDARD_DEVIATION = 150;
	    GRADIENT_THRESHOLD = 30;
	} else if (three.compare(argv[1]) == 0) {
	    imageName = "images/TestImage3.jpg";
	    ACCU_THRESHOLD = 100;
	    CLOSENESS = 0.035;
	    STANDARD_DEVIATION = 140;
	    GRADIENT_THRESHOLD = 20;
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
    rows = image.rows;
    cols = image.cols;
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
    Mat gradient = mySobel(gaussian);
    if (DISPLAY_IMAGE) {
        imshow("Display window", gradient);
	waitKey(0);
    }
    // Thresholding
    Mat edges = myThresholding(gradient, GRADIENT_THRESHOLD);
    if (DISPLAY_IMAGE) {
        imshow("Display window", edges);
	waitKey(0);
    }
    // Hough Transform
    unordered_map<double, vector<int> > lines = myHoughTransform(edges); 

    // Detect Parallelogram
    vector<vector<pair<int, int> > > intersections = detectParallelogram(lines, edges);

    // Display the parallelogram
    displayParallelogram(intersections, image);

    return 0;
}
