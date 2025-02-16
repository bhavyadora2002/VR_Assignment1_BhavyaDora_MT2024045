# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:17:32 2025

@author: MT2024045
"""

import cv2 as cv
import numpy as np

center = cv.imread('C:/Users/Dileep/OneDrive/Desktop/OpenCV/Imgs/center.jpg')
left = cv.imread('C:/Users/Dileep/OneDrive/Desktop/OpenCV/Imgs/left.jpg')
right = cv.imread('C:/Users/Dileep/OneDrive/Desktop/OpenCV/Imgs/right.jpg')
center=cv.add(center,20)
left=cv.add(left,-5)
center=cv.resize(center, (500, 500), interpolation=cv.INTER_AREA)
left=cv.resize(left, (500, 500), interpolation=cv.INTER_AREA)
right=cv.resize(right, (500, 500), interpolation=cv.INTER_AREA)
cv.imshow("left", left)
cv.imshow('center',center)
cv.imshow('right',right)

def stitch_images(img1, img2):
    
    sift = cv.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    center_kp = cv.drawKeypoints(img1, keypoints1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    left_kp = cv.drawKeypoints(img2, keypoints2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("Key Points Image 1", center_kp)
    cv.imshow("Key Points Image 2", left_kp)
    cv.imwrite('C:/Users/Dileep/OneDrive/Desktop/OpenCV/Imgs/keypoints1.jpg', center_kp)
    cv.imwrite('C:/Users/Dileep/OneDrive/Desktop/OpenCV/Imgs/keypoints2.jpg', left_kp)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"Number of good matches: {len(good_matches)}")
    matches_img = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, 
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("Matches", matches_img)
    cv.imwrite('C:/Users/Dileep/OneDrive/Desktop/OpenCV/Imgs/matches.jpg', matches_img)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)    
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    
    height, width, channels = img2.shape
    warped_img1 = cv.warpPerspective(img1, H, (width + img1.shape[1], height)) 
    warped_img1[0:img2.shape[0], 0:img2.shape[1]] = img2

    average_column = ((warped_img1[:, 500] / 2.0 + warped_img1[:, 501] / 2.0)).astype(np.uint8)
    warped_img1[:, 500] = average_column
    warped_img1[:, 501] = average_column    
    blend_width = 5
    for i in range(blend_width):
        alpha = i / blend_width
        warped_img1[:, width - blend_width + i] = (
            warped_img1[:, width - blend_width + i] * (1 - alpha) + warped_img1[:, width + i] * alpha
        ).astype(np.uint8)

    return warped_img1


stitched_12 = stitch_images(center,left)
stitched_123 = stitch_images(right,stitched_12)

cv.imshow("Stitches 1", stitched_12)
cv.imshow("Stitched 2", stitched_123)
cv.imwrite('C:/Users/Dileep/OneDrive/Desktop/OpenCV/Imgs/pano.jpg', stitched_123)

cv.waitKey(0)
cv.destroyAllWindows()