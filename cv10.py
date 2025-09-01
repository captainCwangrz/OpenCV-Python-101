"""
Unit 9: Template Matching & Haar Cascades
- Overview: Detect a template within an image using normalized correlation; then detect faces with a pre-trained Haar cascade.
- Inputs: `ten_of_hearts.png`, `heart_template.png`, `faces.jpg` in the same folder.
- Usage: Press any key in image windows to proceed/close.
"""

import cv2
import numpy as np

'''
Template Matching
- Brute-force: slide the template over the image and score each position.
- Invariant: No built-in scale/rotation invariance; works when size/orientation are stable.
- Methods: SQDIFF, CCORR, CCOEFF (+ _NORMED). For CCOEFF_NORMED, higher is better.
- We use `TM_CCOEFF_NORMED` and threshold strong matches (e.g., >= 0.90).
'''
# --- Template Matching ---
img = cv2.imread('ten_of_hearts.png')
template = cv2.imread('heart_template.png')
h, w = template.shape[:2]

# Compute similarity map and pick high-confidence locations
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
loc = np.where(res >= 0.90)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)

cv2.imshow('Detected', img)
cv2.imshow('Template', template)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Haar Cascade Face Detection
- Uses pre-trained classifiers (Haar features + boosting) for faces, eyes, etc.
- `scaleFactor`: image pyramid step (smaller -> finer search, more compute).
- `minNeighbors`: how many neighboring rectangles are required to keep a detection.
'''
# --- Haar Cascades ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread('faces.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces; tune scaleFactor/minNeighbors for your images
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Unit 9 Summary
Main functions:
 - `cv2.matchTemplate(img, templ, method)` - similarity map
 - `np.where(res >= thr)` - pick candidate locations
 - `cv2.CascadeClassifier(path)` - load Haar cascade
 - `detectMultiScale(gray, scaleFactor, minNeighbors)` - detect faces
 - `cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness)` - draw boxes

Key ideas:
 - Template matching is sensitive to scale/rotation; pick a normalized method and threshold.
 - Haar cascades scan an image pyramid; `scaleFactor` controls granularity, `minNeighbors` prunes false positives.
 - Convert to grayscale for cascades; color is unnecessary for detection.

Tips:
 - If scale varies, generate templates at multiple scales or switch to feature-based methods.
 - Use `TM_SQDIFF(_NORMED)` carefully: lower is better (invert logic).
 - Validate cascade path via `cv2.data.haarcascades`; consider modern detectors (DNN) for robustness.
'''

