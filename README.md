## Parallelograms Detection

**A. Introduction**

The way I detect parallelogram is as follow:

1. First, I convert the colored image into grayscale image using the luminance formula.
2. I normalize the gray scale values to improve contrast. As a result, all 255 values are used.
3. I use Sobel’s edge operator to compute the magnitude values and gradient angles. Then I
    perform non-maxima suppression to reduce the thickness of the edges. In addition, I
    normalize the magnitude values to (0, 255) for display purpose.
4. Then I perform thresholding to filter out edges that don’t have strong enough magnitude
    value. Strong edge pixels are set to 0, and other pixels are set to 255.
5. Next, I use Hough transform on the edge map to detect straight lines. The step size of
    angles and p values are different for each test image. After filling the accumulator using
    the equation of Hough transform, I store theta and p values in a map. The key of the map
    is theta value, and the value of the map is a vector containing all the p values that are
    associated with that theta value.
6. I scan through the map, choose two pairs of parallel lines at a time. Check if their
    intersections correspond to valid edge pixels. If so, we then compute the number of edge
    points (in percentage) that are present for each side. If the percentage is higher than a
    threshold, it is a parallelogram.
7. Finally, I superimpose the detected parallelograms on the original color image.

This is just a simple parallelogram detector, and it is not very robust when dealing with more
complicated image.

**B. Compilation and Run**

The programming language I used is C++.

To run code, you will first need to have OpenCV library installed because that is what I used to
load and display images. More information on how to install OpenCV in this link:
https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html

In the parallelogram/ folder, you can see an executable “parallelogram”. You can execute this
file with an argument to specify which image you want to use. The test images are already
included by default (in the images/ folder). For example, to run the program on second test
image, type:
```
./parallelogram 2
```

You can also specify a file path as the argument.

If you modify the code and want to recompile, type:
```
make
```

If it doesn’t work, you might want to modify the CMakeLists.txt and do:
```
cmake .
```

**3. Example**

Original image
![alt text](https://github.com/shtsai7/parallelogram-detection/blob/master/parallelogram/images/TestImage1c.jpg "Original image")

Normalized gradient magnitude (as a grayscale image)
![alt text](https://github.com/shtsai7/parallelogram-detection/blob/master/parallelogram/images/TestImage1c_gradient.jpg "Gradient magnitude")

Edge map after thresholding. Threshold value used is 30.
![alt text](https://github.com/shtsai7/parallelogram-detection/blob/master/parallelogram/images/TestImage1c_edge.jpg "Edge map")

Original image with the detected parallelograms superimposed on it.
![alt text](https://github.com/shtsai7/parallelogram-detection/blob/master/parallelogram/images/TestImage1c_parallelogram.jpg "Detected parallelogram")

You can see that detected parallelogram does not fits perfectly with the original image. This is because the left side and right side of the parallelogram in the original image are not perfectly parallel to each other. Therefore, I used an angle step size of 8 and p value step size of 6. The displayed result is an approximation of the detected parallelogram.
