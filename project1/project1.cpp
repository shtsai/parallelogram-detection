/**
 *  Computer Vision
 *  Project 1: Parallelogram
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
vector<vector<pair<int, int> > > myHoughTransform(Mat mat);
bool isPointsClose(vector<pair<int,int> >points, double closeness, double SD);
void addLine(Mat mat, int p, float theta, int intensity); 
vector<vector<pair<int, int> > > detectParallelogram(unordered_map<double, vector<int> > lines, vector<pair<int,int> >* points, Mat edges);
vector<pair<int,int> > checkIntersection(double theta1, int a1p1, int a1p2, double theta2, int a2p1, int a2p2, Mat edges, vector<pair<int,int> >* points); 
bool findIntersection (double theta1, int p1, double theta2, int p2, vector<pair<int,int> > &intersection, Mat edges, Mat pointMat, vector<pair<int,int> >* points);
bool validateParallelogram(vector<pair<int, int> > intersection, Mat edges);
void displayParallelogram(vector<vector<pair<int, int> > > intersections, Mat image);

// global variables for detection parameters, initialize in main()
int maxP;
int cols;
int rows;
double ANGLESTEP;
double PSTEP;
int ACCU_THRESHOLD;
double CLOSENESS;
int STANDARD_DEVIATION;
int GRADIENT_THRESHOLD;
double PARALLELOGRAM_THRESHOLD;

/*  This function converst a RGB Mat object into a grayscale Mat.  */
Mat convertToGrayScale(Mat mat)
{
    int channels = mat.channels();
    int nRows = mat.rows;
    int nCols = mat.cols;
    Mat gmat = Mat(nRows, nCols, CV_8UC1, 0.0);
    
    int i, j;
    for (i = 0; i < nRows; i++) {
	for (j = 0; j < nCols; j++) {
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
Mat myEnhancer(Mat mat) {
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
 */
Mat mySobel(Mat mat) {
    int nRows = mat.rows;
    int nCols = mat.cols;
    int minM = INT_MAX;
    int maxM = INT_MIN;
    vector<vector<int> > M (nRows, vector<int>(nCols, 0));	// magnitude
    vector<vector<double> > GA (nRows, vector<double>(nCols, 0));  // gradient angle
    Mat gradient = Mat(nRows, nCols, CV_8UC1, 0.0);
    
    // compute gradient magnitude and gradient angle
    for (int i = 1; i < nRows-1; i++) {
	for (int j = 1; j < nCols-1; j++) {
	    int Gx = mat.at<uchar>(i-1,j+1)+2*mat.at<uchar>(i,j+1)+mat.at<uchar>(i+1,j+1)
	                     -mat.at<uchar>(i-1,j-1)-2*mat.at<uchar>(i,j-1)-mat.at<uchar>(i+1,j-1);
	    int Gy = mat.at<uchar>(i-1,j-1)+2*mat.at<uchar>(i-1,j)+mat.at<uchar>(i-1,j+1)
	                     -mat.at<uchar>(i+1,j-1)-2*mat.at<uchar>(i+1,j)-mat.at<uchar>(i+1,j+1);
	    M.at(i).at(j) = sqrt(pow(Gx, 2) + pow(Gy, 2));
	    GA.at(i).at(j) = atan((double) Gy/Gx) * 180 / PI;
	}
    }

    // Non-maxima suppression (5 x 5 template)
    for (int i = 2; i < nRows-2; i++) {
	for (int j = 2; j < nCols-2; j++) {
	    double angle = GA.at(i).at(j);
	    if (angle < 0) {
		angle += 180;
	    } else if (angle > 180) {
		angle -= 180;
	    }
	    if ((angle >= 0 && angle <= 22.5) || (angle >= 157.5 && angle <= 180)) {
		if (!(M.at(i).at(j) > M.at(i).at(j+1) && M.at(i).at(j) > M.at(i).at(j+2)
		    && M.at(i).at(j) > M.at(i).at(j-1) && M.at(i).at(j) > M.at(i).at(j-2))) {
		    M.at(i).at(j) = 0;
		}	    
	    } else if (angle > 22.5 && angle <= 67.5) {
		if (!(M.at(i).at(j) > M.at(i-1).at(j+1) && M.at(i).at(j) > M.at(i-2).at(j+2)
		    && M.at(i).at(j) > M.at(i+1).at(j-1) && M.at(i).at(j) > M.at(i+2).at(j-2))) {
		    M.at(i).at(j) = 0;
		}	    
	    } else if (angle > 67.5 && angle <= 112.5) {
		if (!(M.at(i).at(j) > M.at(i-1).at(j) && M.at(i).at(j) > M.at(i-2).at(j)
		   && M.at(i).at(j) > M.at(i+1).at(j) && M.at(i).at(j) > M.at(i+2).at(j))) {
		    M.at(i).at(j) = 0;
		}	    
	    } else {
		if (!(M.at(i).at(j) > M.at(i-1).at(j-1) && M.at(i).at(j) > M.at(i-2).at(j-2)
		   && M.at(i).at(j) > M.at(i+1).at(j+1) && M.at(i).at(j) > M.at(i+2).at(j+2))) {
		    M.at(i).at(j) = 0;
		}	    
	    }
	    // update min and max
	    minM = min(minM, M.at(i).at(j));
	    maxM = max(maxM, M.at(i).at(j));
	}
    }

    // normalize magnitude values 
    for (int i = 1; i < nRows-1; i++) {
	for (int j = 1; j < nCols-1; j++) {
	    gradient.at<uchar>(i,j) = ((float) (M.at(i).at(j) - minM) / (float) (maxM-minM)) * 255;
	}
    }
    return gradient;
}

/* This function performs thresholding on the given mat */
Mat myThresholding(Mat mat, int threshold) {
    for (int i = 0; i < mat.rows; i++) {
	for (int j = 0; j < mat.cols; j++) {
	    if (mat.at<uchar>(i,j)  > threshold) {
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
 * This function returns an vector containing vectors of four intersections of parallelograms.
 */

vector<vector<pair<int, int> > > myHoughTransform(Mat mat)
{
    int maxP = sqrt(pow(rows, 2) + pow(cols, 2));
    int psize = maxP * 2 / PSTEP;
    int anglesize = 180/ANGLESTEP;
    int* accumulator = new int[psize * anglesize];
    vector<pair<int,int> >* points = new vector<pair<int, int> >[psize * anglesize];
    unordered_map<double, vector<int> > lines;

    for (int i = 0; i < psize; i++) {	// initialize accumulator to zero
	for (int j = 0; j < anglesize; j++) {
	    accumulator[i * anglesize + j] = 0;
	}
    }
    for (int y = 0; y < mat.rows; y++) {
	for (int x = 0; x < mat.cols; x++) {
	    if (mat.at<uchar>(y, x) == 0) {  // edge point
		for (double theta = ANGLESTEP/2; theta < 180; theta += ANGLESTEP) {
		    double p = x * cos(theta * PI / 180) + y * sin(theta * PI / 180);
		    int pindex = (p + maxP) / PSTEP;
		    int tindex = theta/ANGLESTEP;
		    accumulator[pindex * anglesize + tindex]++;
		    points[pindex * anglesize + tindex].push_back(make_pair(y, x));
		}
	    }
	}
    }

    Mat lineMat = mat.clone();    // a mat used for displaying lines
    for (int i = 0; i < psize; i++) {
	for (int j = 0; j < anglesize; j++) {
	    // check if it is a local maximum
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
		    // skip lines that don't meet the closeness and standard deviation requirement
		    if (!isPointsClose(points[i * anglesize + j], CLOSENESS, STANDARD_DEVIATION)) {
			continue;   
		    }
		    lines[j * ANGLESTEP + ANGLESTEP/2.0].push_back(i * PSTEP - maxP);
		    if (true) {
			addLine(lineMat, i * PSTEP - maxP, j * ANGLESTEP + ANGLESTEP/2, 0);
		    }
		}
	    }
	}
    }
    if (DISPLAY_IMAGE) {
        namedWindow("Display window 2", WINDOW_AUTOSIZE); 
	moveWindow("Display window 2", 20, 20);
        imshow("Display window 2", lineMat);
	waitKey(0);
    }
    
    delete accumulator;

    // detect parallelogram
    vector<vector<pair<int, int> > > intersections = detectParallelogram(lines, points, mat);
    return intersections;
}

/* 
 * Given a list of points, test whether the points are close enough to each other. 
 * Use two measures:
 *	- closness: the percentage of points that are discontinous
 *	- SD: standard deviation of the all points on this line
 */
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
 * Given p and theta value, this function adds a line that satisfies the condition
 * to the input mat.
 */
void addLine(Mat mat, int p, float theta, int intensity)
{
    double COS = cos(theta * PI / 180);
    double TAN = tan(theta * PI / 180);
    for (int y = 0; y < mat.rows; y++) {
	int x = (p / COS) - TAN * y;
	if (x >= 0 && x < mat.cols) {
	    mat.at<uchar>(y, x) = intensity;
	}
    }
}

/*
 * This function detects parallelogram using the provides (theta, p) pairs.
 * We test four lines (l1, l2, l3, l4) in each iterations.
 * l1 and l2 are parallel, and l3 and l4 are parallel. 
 * These two pairs of lines have different theta values.
 * We first compute the intersections of the four lines, and check if the itersections
 * are at edge pixels. Then we validate the parallelograms by going through four sides,
 * and compute the number of edge points (in percentage) that are present for each side.
 * If the percentage is high for each side, then it is a parallelogram.
 */
vector<vector<pair<int, int> > > detectParallelogram(unordered_map<double, vector<int> > lines, vector<pair<int,int> >* points, Mat edges) {   
    // put all angles in an vector
    vector<double> angles;
    for (unordered_map<double, vector<int> >::iterator it = lines.begin(); it != lines.end(); it++) {
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

			    vector<pair<int,int> > intersect = checkIntersection(theta1, a1p1v, a1p2v, theta2, a2p1v, a2p2v, edges, points);
			    if (intersect.size() == 4) {  // find 4 intersections, valid parallelogram
				intersections.push_back(intersect);
			    }
			}
		    }
		}
	    }
	}
    }
    return intersections;
}

/*
 * Given the p values and theta values of four lines, this function checks if their intersections 
 * are near a edge pixel. If we find four such points, we validate whether these points form a 
 * parallelogram in the edge map. 
 * If so, we return a vector containing four points.
 * Otherwise, return an empty vector.
 */
vector<pair<int,int> > checkIntersection(double theta1, int a1p1, int a1p2, double theta2, int a2p1, int a2p2, Mat edges, vector<pair<int,int> >* points) {
    vector<pair<int, int> > intersection;
    Mat pointMat = Mat(rows, cols, CV_8UC1, 255.0);
    
    // clear the vector if findIntersection() returns false
    if (!findIntersection(theta1, a1p1, theta2, a2p1, intersection, edges, pointMat, points)) {
	intersection.clear();
    }
    if (!findIntersection(theta2, a2p1, theta1, a1p2, intersection, edges, pointMat, points)) {
	intersection.clear();
    }
    if (!findIntersection(theta1, a1p2, theta2, a2p2, intersection, edges, pointMat, points)) {
	intersection.clear();
    }
    if (!findIntersection(theta2, a2p2, theta1, a1p1, intersection, edges, pointMat, points)) {
	intersection.clear();
    }

    // validate whether the two intersections forms a parallelogram
    if (intersection.size() == 4 && validateParallelogram(intersection, edges)) {
	return intersection;
    } else {
	intersection.clear();
	return intersection;
    }
}

/*
 * Given the theta values and p values of two lines, this function computes
 * their intersection. And check if the intersection is near an edge pixel.
 * If so, add the point to the list, and return true.
 * Otherwise, return false.
 */
bool findIntersection (double theta1, int p1, double theta2, int p2, vector<pair<int,int> > &intersection, Mat edges, Mat pointMat, vector<pair<int,int> >* points) {
    double i, j;
    int inti, intj;
    i = (cos(theta1*PI/180)*p2 - cos(theta2*PI/180)*p1) 
	/ (sin(theta2*PI/180)*cos(theta1*PI/180)-sin(theta1*PI/180)*cos(theta2*PI/180));
    j = (p1 - i * sin(theta1*PI/180)) / cos(theta1*PI/180);
    if (i < 0 || i >= rows || j < 0 || j >= cols) return false;
    inti = (int) i;
    intj = (int) j;
    bool found = false;

    // check if the intersection is presented in the edge map.
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
	pair<int, int> point (inti, intj);
	intersection.push_back(point);
	return true;
    }
    return false;
}
   
/*
 * We validate the parallelograms by going through four sides,
 * and compute the number of edge points (in percentage) that are present for each side.
 * If the percentage is greater than the given threshold for each side, 
 * then it is a parallelogram.
 */
bool validateParallelogram(vector<pair<int, int> > intersection, Mat edges) {
    int validCount = 0;
    int total = 0;
    for (int p = 0; p < intersection.size(); p++) {
	int localValid = 0;
	int localTotal = 0;
	pair<int,int> point1 = intersection.at(p);
	pair<int,int> point2 = p < intersection.size()-1 ? intersection.at(p+1) : intersection.at(0);
	if (pow(point1.first-point2.first, 2) + pow(point1.second-point2.second, 2) < 100) {
	    // two points are two close
	    return false;
	}

	// generate equation for the line going through these two points
	double m = (double) (point1.first - point2.first) / (point1.second - point2.second);
	double b = (double) point1.first - m * point1.second;
	int jMin = min(point1.second, point2.second);	// get the range of j between two points
	int jMax = max(point1.second, point2.second);
	localTotal += jMax - jMin + 1;
	for (int j = jMin; j <= jMax; j++) {
	    int i = m * j + b;
	    if (i < 0 || i > edges.rows) continue;  // outside of the mat
	    bool found = false;
	    for (int di = -3; di <= 3; di++) {	// check if there is a corresponding edge pixel near this line pixel
		for (int dj = -3; dj <= 3; dj++) {
		    if (edges.at<uchar>(i+di, j+dj) == 0) {
			found = true;
		    }
		}
	    }
	    if (found) localValid++;
	}
	double localScore = (double) localValid / localTotal;
	if (localScore < (PARALLELOGRAM_THRESHOLD)) {
	    return false;
	}
	validCount += localValid;
	total += localTotal;
    }
    double score = (double) validCount / total;
    return score > PARALLELOGRAM_THRESHOLD;
}

/*
 * Given a vector of four points, this function superimpose the the four lines on the given mat.
 * Note that the order of the points matters.
 * The formed lines are (p1->p2, p2->p3, p3->p4, p4->p1).
 */
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
	// the parameters used for each test images is placed here
	String one("1");
	String two("2");
	String three("3");
	if (one.compare(argv[1]) == 0) {
	    imageName = "images/TestImage1c.jpg";
	    ANGLESTEP = 8.0;
	    PSTEP = 6.0;
	    ACCU_THRESHOLD = 100;
	    CLOSENESS = 0.02;
	    STANDARD_DEVIATION = 100;
	    GRADIENT_THRESHOLD = 30;
	    PARALLELOGRAM_THRESHOLD = 0.1;
	} else if (two.compare(argv[1]) == 0) {
	    imageName = "images/TestImage2c.jpg";
	    ANGLESTEP = 5.0;
	    PSTEP = 3.0;
	    ACCU_THRESHOLD = 50;
	    CLOSENESS = 0.08;
	    STANDARD_DEVIATION = 300;
	    GRADIENT_THRESHOLD = 20;
	    PARALLELOGRAM_THRESHOLD = 0.50;
	} else if (three.compare(argv[1]) == 0) {
	    imageName = "images/TestImage3.jpg";
	    ANGLESTEP = 5.0;
	    PSTEP = 3.0;
	    ACCU_THRESHOLD = 20;
	    CLOSENESS = 0.45;
	    STANDARD_DEVIATION = 125;
	    GRADIENT_THRESHOLD = 25;
	    PARALLELOGRAM_THRESHOLD = 0.70;
	} else {   // default configuration
	    imageName = argv[1];	
	    ACCU_THRESHOLD = 20;
	    CLOSENESS = 1;
	    STANDARD_DEVIATION = 1000;
	    GRADIENT_THRESHOLD = 15;
	    PARALLELOGRAM_THRESHOLD = 0.70;
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
    // Apply Sobel operator and display result
    Mat gradient = mySobel(enhanced);
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
    vector<vector<pair<int, int> > > intersections = myHoughTransform(edges); 
    displayParallelogram(intersections, image);

    return 0;
}
