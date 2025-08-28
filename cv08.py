import cv2
import numpy as np

img = cv2.imread('shapes.jpg') # 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# First we do a threshold to get a binary image
_, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

# Apply opening to remove white dots
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cv2.imshow('Original', img)
cv2.imshow('Gray', gray)
cv2.imshow('Threshold', thresh)
cv2.imshow('Opened', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find contours
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f'Number of contours found: {len(contours)}')

# Draw all contours
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3) # img, contours, contourIdx (-1 means all), color, thickness
cv2.imshow('Contours', contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Bounding shapes
bound_shapes = img.copy()
c = max(contours, key=cv2.contourArea) # Runs max() based on contourArea. c is largest contour

# Bounding rect
x, y, w, h = cv2.boundingRect(c)
cv2.rectangle(bound_shapes, (x, y), (x+w, y+h), (255,0,0), 2)

# Minimum area rectangle (rotated)
rect = cv2.minAreaRect(c) # min area bounding rectangle ((cx,cy),(w,h),angle)
box = cv2.boxPoints(rect) # actual corners of the rect
box = box.astype(int) # convert from float to int
cv2.drawContours(bound_shapes, [box], 0, (0,0,255), 2)

# Minimum enclosing circle
(xc, yc), radius = cv2.minEnclosingCircle(c)
cv2.circle(bound_shapes, (int(xc), int(yc)), int(radius), (0,255,255), 2)

cv2.imshow("Largest shape bounded", bound_shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Contour features
# moments are weighted sums of pixel coords - encodes properties like area, centroid, and orientation
# M is a dictionary with keys like "m00, m01, m10, etc.."
M = cv2.moments(c) 

cx = int(M["m10"] / M["m00"]) # m10 first order moments that weight the contour's pixels by x
cy = int(M["m01"] / M["m00"]) # m01 by y, m00 is basically the contour area
cv2.circle(bound_shapes, (cx, cy), 5, (0,0,255), -1)
cv2.imshow("Centroid ", bound_shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()

area = cv2.contourArea(c)
perimeter = cv2.arcLength(c, True)
print("Area:", area, "Perimeter:", perimeter)

# Shape approximation
epsilon = 0.02 * perimeter # 0.001 is considered tight approx, 0.1 is considered loose.
approx = cv2.approxPolyDP(c, epsilon, True)
print(approx)
cv2.drawContours(img, [approx], -1, (0,255,0), 3)
cv2.imshow("Aprrox contour", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Unit 7 Summary
Main Concepts:
  - Contour detection and analysis
  - Shape bounding techniques
  - Contour features and moments
  - Shape approximation
Important functions:
  - cv2.findContours() works with binary images
  - cv2.drawContours() draws contours on an image
  - cv2.boundingRect() computes the bounding box for a contour
  - cv2.minAreaRect() finds the minimum area rectangle for a contour
  - cv2.minEnclosingCircle() finds the minimum enclosing circle for a contour
  - cv2.moments() computes the moments of a contour
  - cv2.approxPolyDP() approximates a contour shape
Typical pipeline:
  1. Preprocessing (e.g., grayscale, thresholding)
  2. Contour detection
  3. Contour analysis (e.g., bounding shapes, features)
  4. Post-processing (e.g., drawing, visualization)
'''