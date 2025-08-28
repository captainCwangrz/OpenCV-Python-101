import cv2
import numpy as np

noisy_img = cv2.imread('noisy_img.png')

# Average blur (box filter)
# Takes the average of all pixels under the kernel area and replaces the central element with this average
blur = cv2.blur(noisy_img, (5, 5)) # kernel size 5x5
cv2.imshow('Noisy Image', noisy_img)
cv2.imshow('Average Blur', blur)

# Gussian blur
# Similar to average blur, but uses a Gaussian kernel which gives more weight to the central pixels
gaussian = cv2.GaussianBlur(noisy_img, (5, 5), 0) # kernel size 5x5, sigmaX=0. 0 means auto-calculate based on kernel size
cv2.imshow('Gaussian Blur', gaussian)


# Median blur
# Takes the median of all pixels under the kernel area and replaces the central element with this median
# Very effective in removing salt-and-pepper noise because median is less sensitive to outliers
median = cv2.medianBlur(noisy_img, 5) # kernel size 5x5
cv2.imshow('Median Blur', median)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Edge detection using Canny
# Gaussian blur helps reduce noise and improve edge detection
# Canny expects a single channel (grayscale) image
edge_img = cv2.imread('edge_img.jpg')
blurred_edge_img = cv2.GaussianBlur(edge_img, (5, 5), 0)
gray = cv2.cvtColor(blurred_edge_img, cv2.COLOR_BGR2GRAY)
edges1 = cv2.Canny(gray, 100, 200) # thresholds
edges2 = cv2.Canny(gray, 150, 250)
edges3 = cv2.Canny(gray, 200, 300)
cv2.imshow('Original Edge Image', edge_img)
cv2.imshow('Thresholds 100-200', edges1)
cv2.imshow('Thresholds 150-250', edges2)
cv2.imshow('Thresholds 200-300', edges3)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Thresholding
_, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) # simple binary thresholding
_, th2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV) # inverse binary thresholding
th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # adaptive thresholding
cv2.imshow('Original Gray Image', gray)
cv2.imshow('Simple Binary Thresholding', th1)
cv2.imshow('Inverse Binary Thresholding', th2)
cv2.imshow('Adaptive Gaussian Thresholding', th3)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Morphological Operations
Kernel - a small shape like a square, circle, or cross. You slide this kernel over the image like a stencil.
The image is binary or grayscale
At each pixel, the kernel defines a neighborhood of pixels to consider

Erosion:
  A pixel survives only if all of its neighbors under the kernel are also white
  - Thin lines become thinner
  - Small white dots vanish
  - Large blobs shrink inward (eroded)
Dilate:
  Vice versa
Opening:
  erosion -> dilation
Closing:
  dilation -> erosion
'''
kernel = np.ones((5,5), np.uint8) # k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) to get various non rectangular shapes
# Erosion: shrink white regions
eroded = cv2.erode(th1, kernel, iterations=1)
# Dilation: expand white regions
dilated = cv2.dilate(th1, kernel, iterations=1)
# Opening: erosion → dilation (removes small noise)
opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
# Closing: dilation → erosion (fills small holes)
closing = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Eroded", eroded)
cv2.imshow("Dilated", dilated)
cv2.imshow("Opening", opening)
cv2.imshow("Closing", closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Unit 4 Summary
Main functions:
 - cv2.blur() - for average blurring
 - cv2.GaussianBlur() - for Gaussian blurring
 - cv2.medianBlur() - for median blurring
 - cv2.Canny() - for edge detection
 - cv2.threshold() - for simple thresholding
 - cv2.adaptiveThreshold() - for adaptive thresholding
 - cv2.erode() - for erosion
 - cv2.dilate() - for dilation
 - cv2.morphologyEx() - for advanced morphological operations (opening, closing, etc.)

Main concepts:
  - Blurring - reducing image noise and detail
  - Edge Detection - identifying significant transitions in intensity
  - Thresholding - segmenting images based on intensity levels
  - Morphological Operations - processing binary images based on shapes
'''