"""
Unit 1: Images & Grayscale
- Overview: Load an image, inspect shape and pixel values, display, convert to grayscale, and save.
- Inputs: `test_img.png` in the same folder.
- Usage: Press any key in image windows to continue/close.
"""

import cv2
from pathlib import Path

print(cv2.__version__ )
#print(cv2.getBuildInformation())

# --- Reading and dimensions ---
img = cv2.imread('test_img.png')
print(f"imread() gives you: {type(img)}")
print(f"img.shape: {img.shape}") # (height, width, color_channel)
img_height = img.shape[0]
img_width = img.shape[1]
img_channel = img.shape[2]
print(f'height: {img_height}, width: {img_width}, channel: {img_channel}')
# First pixel BGR tuple
print(f"First pixel value at img[0,0] (B,G,R): {img[0,0]}")

# Displaying an image
cv2.imshow('Test Image', img)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # closes displayed windows
print()

# Converting to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f'Gray image shape (h x w): {gray_img.shape}')
print(f"First pixel value at gray_img[0,0] (luminance): {gray_img[0,0]}") # grayscale value of the first pixel
print("Standard grayscale conversion formula: Y = 0.299 R + 0.587 G + 0.114 B")
print(f"Calculated grayscale value at img[0,0]: 0.299*{img[0,0][2]} + 0.587*{img[0,0][1]} + 0.114*{img[0,0][0]} = {0.299*img[0,0][2] + 0.587*img[0,0][1] + 0.114*img[0,0][0]}")
cv2.imshow('Gray Image', gray_img)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # closes displayed windows
print()

# Writing an image
cv2.imwrite('gray_img.png', gray_img)

'''
Unit 1 Summary
Main functions:
 - `cv2.imread(path)` - load BGR image as NumPy array
 - `cv2.cvtColor(img, code)` - convert color spaces (e.g., BGR->GRAY)
 - `cv2.imshow(name, img)` / `cv2.waitKey()` - display and pause
 - `cv2.imwrite(path, img)` - save image to disk

Key ideas:
 - Image shape is (height, width, channels); OpenCV uses BGR order.
 - A pixel is a 3-tuple (B, G, R); grayscale is a single channel.
 - `img[y, x]` indexes row (y) then column (x).

Tips:
 - Check `img is not None` after `imread` to avoid NoneType errors.
 - Use small images for quick experiments; large ones slow down UI.
 - Prefer absolute or script-relative paths to avoid CWD surprises.
'''

