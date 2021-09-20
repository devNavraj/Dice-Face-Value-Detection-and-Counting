# Importing the necessary libraries
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read in an image and correct for colorspace
img = cv2.imread("input/dice.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.namedWindow("main", cv2.WINDOW_NORMAL)
cv2.imshow('main', img)
cv2.waitKey(0)

# Create clones and grayscale version of the image
img_clone = img.copy()
img_clone1 = img.copy()

# Create grayscale version of the image
img_clone = cv2.cvtColor(img_clone, cv2.COLOR_RGB2GRAY)

# Apply Gaussian blurring order to soften image
img_clone = cv2.GaussianBlur(img_clone, (5, 5), 0)

# Threshold the image
ret, img_thresh = cv2.threshold(img_clone, 207, 255, cv2.THRESH_BINARY)

# Use Canny Edge Detection
img_canny = cv2.Canny(img_thresh, 2, 2*2, 5)

def calculate_value(roi):
    """ This function takes in the Region of Interests and 
    returns the total number of RoIs."""
    # Set up the detector with default parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)
    key_points = detector.detect(roi)
    
    return len(key_points)

# Get the contours of the dice
contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    # Find the area
    contours_area = cv2.contourArea(contours[i])
    
    # Get the region of interest (for the particular die)
    x, y, w, h = cv2.boundingRect(contours[i])
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
    dice_roi = img_clone1[y: (y + h), x: (x + w)]
    
    # Calculate the number of RoIs
    number = calculate_value(dice_roi)
    
    # Label the dice with its number of dots
    img = cv2.putText(img, str(number), (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (0, 255, 0), 5)

# Label the total number of dots
img = cv2.putText(img, "Total: " + str(number), (25, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (0, 255, 0), 5)
cv2.drawContours(img, contours, -1, (0, 0, 255), 5)

# Save the result
outputName = 'output_dice.png'
outputPath = "./output"
cv2.imwrite(os.path.join(outputPath,outputName), img)

cv2.imshow('main', img)
cv2.waitKey(0)
