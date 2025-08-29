"""
Unit 2: Drawing & Basic Ops
- Overview: Draw primitives and text, crop, resize, flip, rotate, and reuse ROIs.
- Inputs: `gray_img.png` (from Unit 1) in the same folder.
- Usage: Press any key in image windows to proceed/close.
"""

import cv2

gray_img = cv2.imread('gray_img.png')

# --- Drawing on images ---
drawn_image = gray_img.copy() # deep copy
# Line: img, start, end, color, thickness (0 because grayscale)
cv2.line(drawn_image, (0, drawn_image.shape[0]//2), (drawn_image.shape[1], drawn_image.shape[0]//2), 0, 5) 
# Rectangle: img, top-left, bottom-right, color, thickness
cv2.rectangle(drawn_image, (drawn_image.shape[1]//4, drawn_image.shape[0]//4), (drawn_image.shape[1]*3//4, drawn_image.shape[0]*3//4), 0, 5)
# Circle: img, center, radius, color, thickness
cv2.circle(drawn_image, (drawn_image.shape[1]//2, drawn_image.shape[0]//2), drawn_image.shape[0]//8, 0, -1) #-1 thickness fills the circle
# Text: img, text, bottom-left, font, font-scale, color, thickness
cv2.putText(drawn_image, 'Hello OpenCV', (drawn_image.shape[1]//4, drawn_image.shape[0]//8), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
cv2.imshow("drawings", drawn_image)
cv2.imwrite('drawn_image.png', drawn_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print()

# --- Cropping images ---
cropped_img = gray_img.copy()
cropped_img = cropped_img[0:cropped_img.shape[0]//2,0:] #rows, cols
print(f'cropped_img shape: {cropped_img.shape}')
cv2.imshow("cropped", cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print()

# --- Resizing images ---
resized_img = gray_img.copy()
# resize(img, (width, height), interpolation=), default is INTER_AREA for shrinking and INTER_LINEAR for enlarging
# enlarging methods: INTER_CUBIC (better but slower), INTER_LINEAR (faster)
# shrinking methods: INTER_AREA (better but slower), INTER_LINEAR (faster)
resized_img = cv2.resize(resized_img, (resized_img.shape[1]//2, resized_img.shape[0]//2))
print(f'resized_img shape: {resized_img.shape}')
cv2.imshow("resized_img", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print()

# --- Flipping images ---
flipped_img = gray_img.copy()
flipped_h = cv2.flip(flipped_img, 1) #1: horizontal, -1: both
flipped_v = cv2.flip(flipped_img, 0) #0: vertical
flipped_hv = cv2.flip(flipped_img, -1) #-1: both
cv2.imshow("flipped_h", flipped_h)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("flipped_v", flipped_v)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("flipped_hv", flipped_hv)
cv2.waitKey(0)
cv2.destroyAllWindows()
print()

# --- Rotating images ---
rotated_img = gray_img.copy()
rotated_img = cv2.rotate(rotated_img, cv2.ROTATE_90_CLOCKWISE) #ROTATE_90_COUNTERCLOCKWISE, ROTATE_180
cv2.imshow("rotated_img", rotated_img)
cv2.waitKey(0) 
cv2.destroyAllWindows()
print()

# --- ROI (reuse) ---
roi_repeated = gray_img.copy()
cv2.circle(roi_repeated, (30, 30), 30, 255, 1)
roi = roi_repeated[0:60, 0:60]
for x in range(0, roi_repeated.shape[1], 60):
  for y in range(0, roi_repeated.shape[0], 60):
    # Checkerboard pattern
    if (x//60 + y//60) % 2 == 0:
      roi_repeated[y:y+60, x:x+60] = roi
cv2.imshow("roi_repeated", roi_repeated)
cv2.waitKey(0)
cv2.destroyAllWindows()
print()

'''
Unit 2 Summary
Main functions:
 - `cv2.line(img, pt1, pt2, color, thickness)` - draw line
 - `cv2.rectangle(img, pt1, pt2, color, thickness)` - draw rectangle
 - `cv2.circle(img, center, radius, color, thickness)` - draw circle
 - `cv2.putText(img, text, org, font, scale, color, thickness)` - text
 - `cv2.resize(img, (w, h), interpolation)` - resize
 - `cv2.flip(img, code)` - flip (0=vertical, 1=horizontal, -1=both)
 - `cv2.rotate(img, code)` - rotate (e.g., ROTATE_90_CLOCKWISE)

Key ideas:
 - Coordinate order is (x, y) for drawing points; array indexing is [y, x].
 - Grayscale images use a single channel; color is BGR.
 - ROI slicing creates a view; assign back to copy regions efficiently.

Tips:
 - Use integer coordinates and odd thicknesses for crisp lines.
 - Choose interpolation: INTER_AREA (shrink), INTER_LINEAR (fast), INTER_CUBIC (quality).
 - Keep UI snappy by closing windows (`waitKey` + `destroyAllWindows`).
'''

