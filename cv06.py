"""
Unit 5: Geometric Transforms
- Overview: Apply translation, rotation, affine, and perspective transforms with short math notes.
- Inputs: `test_img.png` in the same folder.
- Usage: Press any key in image windows to proceed/close.
"""

import cv2
import numpy as np

img = cv2.imread('test_img.png')
h, w = img.shape[:2]


'''
Translation Matrix
M = np.float32([[1, 0, tx], [0, 1, ty]])
 x  [ 1 0 tx ]  => 1x + 0y + tx => x + tx 
 y  [ 0 1 ty ]  => 0x + 1y + ty => y + ty

 Technically, we should use 3x3 homogeneous matrix for affine transformation.
 x   [ 1 0 tx ]   x + tx
 y * [ 0 1 ty ] = y + ty
 1   [ 0 0  1 ]     1
 But warping function in OpenCV takes 2x3 matrix affine transformations.
'''
M = np.float32([[1, 0, 100], [0, 1, 50]]) # warpAffine expects float32
shifted = cv2.warpAffine(img, M, (w, h)) # note the output size is width by height not height by width like img.shape

cv2.imshow('Original Image', img)
cv2.imshow('Shifted Image', shifted)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Rotation Matrix (origin)
M = cosθ -sinθ
    sinθ cosθ 

x' = xcosθ  - ysinθ 
y' = xsinθ  + ycosθ

Think unit circle
x = rcosθ
x' = rcos(θ + φ) = r(cosθcosφ - sinθsinφ) = xcosφ - ysinφ
y' = rsin(θ + φ) = r(sinθcosφ + cosθsinφ) = xsinφ + ycosφ
Intuitively, xcosφ - ysinφ means we are reducing x by sinφ portion of y and adding cosφ portion of x.

cv2 provides a function to generate rotation matrix around a point (cx, cy)
'''
center = (w // 2, h // 2)
angle = 45 # counter-clockwise
scale = 1.0
M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(img, M, (w, h))
cv2.imshow('Rotated Image', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Affine Transformation
Affine = rotation + translation + scaling + shearing
Affine transformation preserves lines and parallelism (but not necessarily distances and angles).
It's essentially a linear transformation followed by a translation. (combination of above + more)

Matrix is still 
a c tx
b d ty

x' = ax + cy + tx
y' = bx + dy + ty

cv2 just needs 3 pairs of corresponding points to determine the 6 unknowns. (3 pairs must not be collinear)
'''
pts1 = np.float32([[50, 50], [200, 50], [50, 200]]) # original 'triangle'
pts2 = np.float32([[10, 100], [200, 50], [100, 250]]) # warped 'triangle'
M = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(img, M, (w, h))
cv2.imshow('Affine Transform', affine)
cv2.waitKey(0)
cv2.destroyAllWindows() 

'''
Perspective Transformation
Perspective lets you change the perspective of the image. It can be considered a more generalized form of affine transformation.
For example, it can 'unskew' something like a tilted piece of paper.
Needs 4 pairs of corresponding points to determine the 8 unknowns.
'''
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv2.getPerspectiveTransform(pts1, pts2)
perspective = cv2.warpPerspective(img, M, (300, 300)) # 300x300 output because pts2 is within that range ?
cv2.imshow('Perspective Transform', perspective)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Unit 5 Summary 
Main functions:
 - `cv2.warpAffine(img, M, (w, h))` — translation/rotation/affine
 - `cv2.warpPerspective(img, M, (w, h))` — perspective warp
 - `cv2.getRotationMatrix2D(center, angle, scale)` — rotation matrix
 - `cv2.getAffineTransform(pts1, pts2)` — 2×3 affine matrix
 - `cv2.getPerspectiveTransform(pts1, pts2)` — 3×3 homography

Key ideas:
 - Affine preserves parallel lines; perspective can map quadrilaterals.
 - 2×3 affine matrices model rotation, scale, shear + translation.
 - Homographies (3×3) work in homogeneous coordinates; enable “unskewing”.

Tips:
 - Use float32 matrices; point sets must be non‑collinear.
 - Choose output size carefully to avoid cropping after transforms.
 - Compose transforms by multiplying matrices before warping.
'''
