import cv2

gray_img = cv2.imread('gray_img.png')

#Drawing on images
drawn_image = gray_img.copy() #deep copy
#line: img, start, end, color, thickness (color 0 cuz of greyscale)
cv2.line(drawn_image, (0, drawn_image.shape[0]//2), (drawn_image.shape[1], drawn_image.shape[0]//2), 0, 5) 
#rectangle: img, top-left, bottom-right, color, thickness
cv2.rectangle(drawn_image, (drawn_image.shape[1]//4, drawn_image.shape[0]//4), (drawn_image.shape[1]*3//4, drawn_image.shape[0]*3//4), 0, 5)
#circle: img, center, radius, color, thickness
cv2.circle(drawn_image, (drawn_image.shape[1]//2, drawn_image.shape[0]//2), drawn_image.shape[0]//8, 0, -1) #-1 thickness fills the circle
#text: img, text, bottom-left, font, font-scale, color, thickness
cv2.putText(drawn_image, 'Hello OpenCV', (drawn_image.shape[1]//4, drawn_image.shape[0]//8), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
cv2.imshow("drawings", drawn_image)
cv2.imwrite('drawn_image.png', drawn_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print()

#Cropping images
cropped_img = gray_img.copy()
cropped_img = cropped_img[0:cropped_img.shape[0]//2,0:] #rows, cols
print(f'cropped_img shape: {cropped_img.shape}')
cv2.imshow("cropped", cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print()

#Resizing images
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

#Flipping images
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

#Rotating images
rotated_img = gray_img.copy()
rotated_img = cv2.rotate(rotated_img, cv2.ROTATE_90_CLOCKWISE) #ROTATE_90_COUNTERCLOCKWISE, ROTATE_180
cv2.imshow("rotated_img", rotated_img)
cv2.waitKey(0) 
cv2.destroyAllWindows()
print()

#ROI
roi_repeated = gray_img.copy()
cv2.circle(roi_repeated, (30, 30), 30, 255, 1)
roi = roi_repeated[0:60, 0:60]
for x in range(0, roi_repeated.shape[1], 60):
  for y in range(0, roi_repeated.shape[0], 60):
    if (x//60 + y//60) % 2 == 0: # checkerboard pattern
      roi_repeated[y:y+60, x:x+60] = roi
cv2.imshow("roi_repeated", roi_repeated)
cv2.waitKey(0)
cv2.destroyAllWindows()
print()

'''
Unit 2 Summary
Drawing:
  - cv2.line(img, start coord, end coord, color, thickness)
  - cv2.rectangle(img, top-left coord, bottom-right coord, color, thickness))
  - cv2.circle(img, center coord, radius, color, thichkness)
  - cv2.putText(img, text, bottom-left coord, font, font-scale, color, thickness)
Cropping:
  - img[row_start:row_end, col_start:col_end ]
Resizing:
  - cv2.resize(img, (new_width, new_height), interpolation)
Flipping:
  - cv2.flip(img, code) # code: 0 vertical, 1 horizontal, -1 both
Rotating:
  - cv2.rotate(img, code) # code: ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE, ROTATE_180
ROI:
  - roi = img[y1:y2, x1:x2]
  - img[y:y+h, x:x+w] = roi
'''

