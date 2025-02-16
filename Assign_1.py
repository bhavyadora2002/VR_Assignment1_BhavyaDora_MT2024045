# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:19:30 2025

@author: MT2024045
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/Dileep/OneDrive/Desktop/OpenCV/Imgs/coin.png')
image=cv.resize(image, (500, 500), interpolation=cv.INTER_AREA)
#Converting to grayscale
gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
#Blur
blurred = cv.GaussianBlur(gray, (7, 7), 2)
#Canny Edge Detection
edges = cv.Canny(blurred, 50, 150)
cv.imshow('Blurred Image',blurred)
cv.imshow('Canny Edge',edges)
cv.imwrite('C:/Users/Dileep/OneDrive/Desktop/OpenCV/Imgs/canny.jpg', edges)

#Counter plotting
contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contour = image.copy()
cv.drawContours(contour, contours, -1, (0, 255, 0), 2)  
cv.imshow("Contour Detection", contour)
cv.imwrite('C:/Users/Dileep/OneDrive/Desktop/OpenCV/Imgs/contour.jpg', contour)

#SEGMENTATION
#Filling contours
cv.drawContours(contour, contours, -1, (0, 255, 0), thickness=cv.FILLED)
cv.imshow("Contour Segmentation", contour)
cv.imwrite('C:/Users/Dileep/OneDrive/Desktop/OpenCV/Imgs/contour_seg.jpg', contour)

#Binary Segmentation
clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(blurred)
cv.imshow("Equi", clahe_image)
_, binary = cv.threshold(blurred, 150, 255, cv.THRESH_BINARY)
cv.imshow(" Binary Image",binary )
cv.imwrite('C:/Users/Dileep/OneDrive/Desktop/OpenCV/Imgs/binary_seg.jpg', binary)

#Coloured Segmentation
colored_coins = np.zeros_like(image)
for i, contour in enumerate(contours):
    color = np.random.randint(0, 255, size=(3,)).tolist() 
    cv.drawContours(colored_coins, [contour], -1, color, thickness=cv.FILLED)
colored = cv.addWeighted(image, 0.6, colored_coins, 0.4, 0)
cv.imshow(" Coloured Image",colored )
cv.imwrite('C:/Users/Dileep/OneDrive/Desktop/OpenCV/Imgs/color_seg.jpg', colored)

#Segmenting individual coin
for i, contour in enumerate(contours):
    mask = np.zeros_like(gray)
    cv.drawContours(mask, [contour], -1, 255, -1)
    coin = cv.bitwise_and(image, image, mask=mask)
    coin=cv.resize(coin, (200, 200), interpolation=cv.INTER_AREA)
    cv.imshow(f'Segmented Coin {i+1}', coin)  
    save_path = f'C:/Users/Dileep/OneDrive/Desktop/OpenCV/Imgs/SegmentedCoin_{i+1}.jpg'
    cv.imwrite(save_path, coin)
    plt.figure()
    plt.title(f'Segmented Coin {i+1}')
    plt.axis('off')
    plt.imshow(coin, cmap='gray')
    plt.show()

#Number of coins detection
def coin_count(contours):
    return len(contours)
print("Number Coins in the image are",coin_count(contours))
cv.waitKey(0)
cv.destroyAllWindows()

