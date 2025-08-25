import cv2
from pathlib import Path

print(cv2.__version__ )
#print(cv2.getBuildInformation())

#Reading an image and its dimensions
img = cv2.imread('test_img.png')
print(f"imread() gives you: {type(img)}")
print(f"img.shape: {img.shape}") # (height, width, color_channel)
img_height = img.shape[0]
img_width = img.shape[1]
img_channel = img.shape[2]
print(f'height: {img_height}, width: {img_width}, channel: {img_channel}')
print(f"First pixel value at img[0,0] (B,R,G): {img[0,0]}") # BGR values of the first pixel

#Displaying an image
cv2.imshow('Test Image', img)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # closes displayed windows
print()

#Converting to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f'Gray image shape (h x w): {gray_img.shape}')
print(f"First pixel value at gray_img[0,0] (luminance): {gray_img[0,0]}") # grayscale value of the first pixel
print("Standard grayscale conversion formula: Y = 0.299 R + 0.587 G + 0.114 B")
print(f"Calculated grayscale value at img[0,0]: 0.299*{img[0,0][2]} + 0.587*{img[0,0][1]} + 0.114*{img[0,0][0]} = {0.299*img[0,0][2] + 0.587*img[0,0][1] + 0.114*img[0,0][0]}")
cv2.imshow('Gray Image', gray_img)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # closes displayed windows
print()

#Writing an image
cv2.imwrite('gray_img.png', gray_img)

'''
Unit 1 Summary
img is a numpy array: width x height x color_channel
ex. 1080 x 1920 x 3
img[0] gives you the first row of pixels
img[0].shape is (1920, 3)
img[0,0] then gives you the first pixel in that row
len(img) == img.shape[0] == img_height == 1080
================================================================================================================================================================================
'''

