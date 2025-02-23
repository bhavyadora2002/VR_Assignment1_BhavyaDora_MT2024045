# VR_Assignment1_BhavyaDora_MT2024045
## Part 1: Use computer vision techniques to Detect, segment, and count coins from an image containing scattered Indian coins.
## BRIEF ABOUT CODE
* DETECTION:
  * Detected edges by using Canny Edge Detection after performing some preprocessing by converting the image to gray scale image and blurring it by using gaussian blur.
  * Plotted contours around the edges of coins.
* SEGMENTATION:
  * Filled the obtained contours with a unique colour to identify coins.
  * Tried binary segmentation, which did not work well as the background of my image and some parts of the coins have same intensity.
  * Assigned different colour to each coin, by creating a black canvas, looping through detected contours, assigning random colours followed by blending with original image.
  * Extracted individual segmented coins by using a mask for each contour and applying bitwise operators to isolate them.
* COUNTING:
  * Counted number of contours to get number of coins.
## STEPS TO RUN
1. Ensure you have OpenCV installed:
  ```sh
  pip install opencv-python numpy
```
2. Make sure to update the input image path in the code as per its location.
3. Open the terminal, navigate to the script location, and run.
  ```sh
  python Assign_1.py
```
## INPUT AND OUTPUT FILES
* Input Image - coin.png
* Canny Edge Detection - canny.jpg
* Contour Detection - contour.jpg
* Contour Segmentation - contour_seg.jpg
* Binary Segmentation - binary_seg.jpg
* Color Segmentation - color_seg.jpg
* Segmented Coins - SegmentedCoin_1.jpg to SegmentedCoin_18.jpg

## Part 2: Create a stitched panorama from multiple overlapping images.
## BRIEF ABOUT CODE
* Loaded 3 images and converted them to grey scale.
* Detected key points using SIFT algorithm.
* Detected matched features using the FLANN-based matcher in OpenCV.
* Used RANSAC to eliminate outliers.
* Found the transformation (homography) between the two images to align them properly for stitching.
* Warped images using computed homography matrix.
* Used weighted average technique for blending the warped images.
## STEPS TO RUN
1. Ensure you have OpenCV installed:
  ```sh
  pip install opencv-python numpy
```
2. Make sure to update the input image path in the code as per its location.
3. Open the terminal, navigate to the script location, and run.
  ```sh
  python Assign_2.py
```
## INPUT AND OUTPUT FILES
* Input Images - left.jpg,center.jpg,right.jpg
* Keypoints - keypoints1.jpg,keypoints2.jpg
* Matches - matches.jpg
* Panoroma - stitched_123.jpg
* Pano_set_2 in Imgs contains another example of panoroma
  
