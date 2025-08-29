"""
Unit 8: Corner Detection
- Overview: Harris and Shi-Tomasi corners, ORB keypoints, and feature matching.
- Inputs: `chessboard.png`, `speaker1.jpg`, `speaker2.jpg` in the same folder.
- Usage: Press any key in image windows to proceed/close.
"""

import cv2
import numpy as np

img = cv2.imread('chessboard.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
Corner Harris
"If you shift the local image patch a little bit in x or y, how much does the intensity change?"
  - Edges: small change along the edge, big change across the edge
  - Flat regions: small change in all directions
  - Corners: big change in all directions
'''
gray = np.float32(gray) # cornerHarris expects float32
# Harris params:
# - blockSize: neighborhood size
# - ksize: Sobel kernel size
# - k in [0.04, 0.06]: sensitivity to corners
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.06)
# dst is grayscale corner response map. same w and h as input. float32 values.
# large positive -> strong corners
# negative -> edges or flat areas
# small positive -> background or flat regions
dst = cv2.dilate(dst, None) # dilate to enhance corner points. None means default 3x3 kernel and iteration = 1
max_response = dst.max() # this is numpy function for max
threshold = 0.01 * max_response # this is a common thresholding strategy 1% of max
mask = dst > threshold # numpy boolean mask. array of True/False based on threshold.
img[mask] = [0, 0, 255] # set those pixels to red in original image. numpy boolean indexing in action.
cv2.imshow('Harris Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Shi-Tomasi Corner Detection (better)
'''
img = cv2.imread('chessboard.png')
# goodFeaturesToTrack params:
# - maxCorners: maximum number of corners
# - qualityLevel: minimal accepted corner quality (lower finds more)
# - minDistance: minimum distance between corners
corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.95, minDistance=10)
for c in corners:
    x, y = c.ravel()
    cv2.circle(img, (int(x), int(y)), 4, (255, 0, 0), -1)
cv2.imshow('Shi-Tomasi Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Keypoints with ORB
'''
img = cv2.imread('chessboard.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ORB_create params:
# - nfeatures: maximum number of features to retain
# - fastThreshold: threshold for the FAST feature detector
# - nlevels: number of pyramid levels
# - scaleFactor: scale factor between levels
orb = cv2.ORB_create()
# Detect keypoints and compute descriptors
kp, des = orb.detectAndCompute(gray, None)
# Draw keypoints
img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
cv2.imshow('ORB Keypoints', img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Feature Matching
'''
img1 = cv2.imread('speaker1.jpg', 0) # 0 means grayscale
img2 = cv2.imread('speaker2.jpg', 0)

img1 = cv2.resize(img1, (768, 1080))
img2 = cv2.resize(img2, (768, 1080))

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
cv2.imshow('Feature Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Unit 8 Summary
Main functions:
 - `cv2.cornerHarris(gray32f, block, ksize, k)` - Harris response
 - `cv2.dilate(img, kernel)` - emphasize peaks
 - `cv2.goodFeaturesToTrack(gray, max, quality, minDist)` - Shi-Tomasi
 - `cv2.ORB_create()` / `orb.detectAndCompute(gray, None)` - ORB keypoints + descriptors
 - `cv2.drawKeypoints(img, kp, None, color, flags)` - visualize keypoints
 - `cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck)` - brute-force matcher for ORB
 - `bf.match(d1, d2)` / `cv2.drawMatches(...)` - match and draw top-N

Key ideas:
 - Harris measures intensity change for small shifts; corners respond strongly.
 - Shi-Tomasi selects points with large minimum eigenvalue (often cleaner).
 - ORB = FAST keypoints + binary (BRIEF-like) descriptors; match by Hamming distance.
 - Sort matches by distance (lower is better); optionally filter further.

Tips:
 - Convert to grayscale for detectors/descriptors; use float32 for Harris.
 - Tune `qualityLevel` (e.g., 0.01-0.1) and `minDistance` for Shi-Tomasi density.
 - For matching, consider `knnMatch` + Lowe's ratio test and enable `crossCheck` for stricter matches.
 - Ensure input images exist and are comparable in scale/content; resize thoughtfully.
'''
