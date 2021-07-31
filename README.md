# Dice-Face-Value-Detection-and-Counting

I used the Python programming language and its OpenCV library for Computer Vision implementation.

The steps I used:
- Converted image to its Grayscale version
- Applied Gaussian blurring order to soften image
- Applied threshold to reduce the information of the image
- To find the boundaries, I used Canny Edge Detection Algorithm
- Found contours of the prepared image.
- Created a Region of Interest using the contours area.
- Applied Blob Detector to find keypoints and find the face value of dice. (There will never be a perfect dot. So, I defined minimum inertia 0.5.)

I have implemented my code in both interactive python notebook i.e ipynb file and python file. 
