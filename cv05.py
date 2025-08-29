"""
Unit 4: Blurring, Edges, Thresholds, Morphology
- Overview: Compare blur methods, run Canny edge detection, apply thresholding, and demonstrate morphological ops.
- Inputs: `noisy_img.png`, `edge_img.jpg` in the same folder.
- Usage: Press any key in image windows to proceed/close.
"""

import cv2
import numpy as np

# --- Blurring (Average, Gaussian, Median) ---
noisy_img = cv2.imread('noisy_img.png')

# Average blur (box filter): replace center with neighborhood average
# - ksize: odd tuple (w, h). Larger -> more smoothing, more detail loss.
blur = cv2.blur(noisy_img, (5, 5))
cv2.imshow('Noisy Image', noisy_img)
cv2.imshow('Average Blur', blur)

# Gaussian blur: weighted average emphasizing center
# - ksize: odd tuple; - sigmaX=0 lets OpenCV choose based on kernel size.
gaussian = cv2.GaussianBlur(noisy_img, (5, 5), 0)
cv2.imshow('Gaussian Blur', gaussian)

# Median blur: non-linear; robust to salt-and-pepper while preserving edges
# - ksize: odd integer window size.
median = cv2.medianBlur(noisy_img, 5)
cv2.imshow('Median Blur', median)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Edge Detection (Canny) ---
edge_img = cv2.imread('edge_img.jpg')
blurred_edge_img = cv2.GaussianBlur(edge_img, (5, 5), 0)
gray = cv2.cvtColor(blurred_edge_img, cv2.COLOR_BGR2GRAY)

# Hysteresis thresholds: (low, high). Tune based on image contrast and noise.
edges1 = cv2.Canny(gray, 100, 200)
edges2 = cv2.Canny(gray, 150, 250)
edges3 = cv2.Canny(gray, 200, 300)
cv2.imshow('Original Edge Image', edge_img)
cv2.imshow('Thresholds 100-200', edges1)
cv2.imshow('Thresholds 150-250', edges2)
cv2.imshow('Thresholds 200-300', edges3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Thresholding (binary, inverse, adaptive) ---
# Simple global thresholding: threshold, maxval, type
_, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, th2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Adaptive Gaussian: chooses local threshold per neighborhood
# - blockSize: odd neighborhood size; - C: constant subtracted from mean.
th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 2)
cv2.imshow('Original Gray Image', gray)
cv2.imshow('Simple Binary Thresholding', th1)
cv2.imshow('Inverse Binary Thresholding', th2)
cv2.imshow('Adaptive Gaussian Thresholding', th3)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Morphological Operations
Kernel: small shape (e.g., square/circle/cross) that defines a neighborhood.
Image: typically binary (foreground=white); operations examine neighborhoods per pixel.

Erosion:
  Pixel stays white only if all neighbors under kernel are white
  - Thin lines get thinner
  - Small white dots vanish
  - Large blobs shrink inward (eroded)
Dilation:
  Opposite of erosion (expands white regions)
Opening:
  erosion -> dilation (removes small white noise)
Closing:
  dilation -> erosion (fills small black holes)
'''
kernel = np.ones((5, 5), np.uint8)
# Tip: try non-rect kernels
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# Erosion: shrink white regions
eroded = cv2.erode(th1, kernel, iterations=1)

# Dilation: expand white regions
dilated = cv2.dilate(th1, kernel, iterations=1)

# Opening: erosion -> dilation (removes small noise)
opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)

# Closing: dilation -> erosion (fills small holes)
closing = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Eroded', eroded)
cv2.imshow('Dilated', dilated)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Unit 4 Summary
Main functions:
 - `cv2.blur()` - average blurring
 - `cv2.GaussianBlur()` - Gaussian blurring
 - `cv2.medianBlur()` - median blurring
 - `cv2.Canny()` - edge detection
 - `cv2.threshold()` - simple thresholding
 - `cv2.adaptiveThreshold()` - adaptive thresholding
 - `cv2.erode()` - erosion
 - `cv2.dilate()` - dilation
 - `cv2.morphologyEx()` - opening/closing

Key ideas:
 - Blur ksize: odd sizes (3,5,7); larger smooths more but blurs edges.
 - Canny thresholds: scale with contrast; blur first to reduce noise.
 - Thresholding: global T vs. adaptive block/C for uneven lighting.
 - Morphology: kernel shape/size and iteration count tailor cleanup.

Tips:
 - Convert to grayscale before Canny/thresholding.
 - Use adaptive thresholding under varying illumination.
 - Prefer binary images (foreground=white) for morphology semantics.
 - Try cv2.getStructuringElement for ellipse/cross kernels when shapes matter.
'''

