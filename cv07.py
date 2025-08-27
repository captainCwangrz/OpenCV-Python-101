'''
HSV Color Space
========================================
Hue = the base color (0-179 in OpenCV)
  Red ~= 0-10 or 170-180
  Green ~= 35-85
  Blue ~= 100-130
Saturation = how 'pure' the color is (0 = grayish, 255 = fully colored)
Value = brightness (0 = dark, 255 = bright)
'''

import cv2

# Convert BGR to HSV
img = cv2.imread('tree_img.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range of green color in HSV
lower_green = (20, 60, 45) # HSV lower bound
upper_green = (70, 255, 255) # HSV upper bound

mask = cv2.inRange(hsv, lower_green, upper_green) # Masks are binary images. Pixels in range are 255, everything else is 0.

# Applying the mask to the original image to keep only the selected colors
result = cv2.bitwise_and(img, img, mask=mask)

# Function to display HSV values on mouse hover
# Easier to pick colors in HSV space this way
def show_hsv(event, x, y, flags, param):
  if event == cv2.EVENT_MOUSEMOVE:
    pixel = hsv[y, x]
    h, s, v = pixel
    print(f'HSV: ({h}, {s}, {v})')

cv2.imshow('Original', img)
cv2.imshow('Mask', mask)
cv2.imshow('Result', result)
cv2.setMouseCallback('Original', show_hsv) # Register mouse event handler
cv2.waitKey(0)
cv2.destroyAllWindows()


# Apply color tracking to video
cap = cv2.VideoCapture(0)
while True:
  ret, frame = cap.read()
  if not ret:
    print("Failed to grab frame")
    break
  
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  
  # Red has two ranges in HSV
  lower_red1 = (0, 110, 50)
  upper_red1 = (15, 255, 255)
  lower_red2 = (170, 110, 50)
  upper_red2 = (180, 255, 255)

  mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
  mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
  mask = cv2.bitwise_or(mask1, mask2) # Fuse the two masks

  result = cv2.bitwise_and(frame, frame, mask=mask)

  cv2.imshow('Frame', frame)
  cv2.imshow('Mask', mask)
  cv2.imshow('Result', result)
  cv2.setMouseCallback('Frame', show_hsv)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()

'''
Unit 6 Summary
Main Concepts:
- HSV
- Color detection and tracking
- Bitwise operations
- Mouse events

For tracking:
 - Avoid the trap of aiming for a perfect mask
 - User morpholoty, countour tracking, Kalman filters, etc. to improve results
'''