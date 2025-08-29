"""
Unit 3: Record Grayscale Video
- Overview: Capture webcam, flip horizontally, convert to grayscale, and write MP4.
- Inputs: Default camera index 0.
- Usage: Press 'q' in the preview window to stop recording.
"""

import cv2

cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# FourCC and writer params
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('private_output.mp4', fourcc, fps, (w, h), isColor=False)
while True:
  ret, frame = cap.read()
  if not ret:
    print("Failed to grab frame")
    break
  frame = cv2.flip(frame, 1)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  out.write(frame)
  cv2.imshow("Recording...", frame)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
out.release()
cv2.destroyAllWindows()

'''
Unit 3 Summary
Main functions:
 - `cv2.VideoWriter_fourcc(*'mp4v')` - MP4 codec
 - `cv2.VideoWriter(path, fourcc, fps, (w, h), isColor)` - writer
 - `out.write(frame)` - append a frame
 - `cv2.flip(img, 1)` - horizontal flip
 - `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` - grayscale conversion
 - `out.release()` / `cv2.destroyAllWindows()` - cleanup

Key ideas:
 - FPS drives playback speed; an incorrect FPS makes video too fast/slow.
 - Frame size must match the writer's (w, h) exactly.
 - `isColor=False` expects single-channel frames (e.g., grayscale).

Tips:
 - If `cap.get(FPS)` returns 0, choose a sane default (e.g., 30).
 - Verify writer opened via `out.isOpened()` to catch codec issues.
 - Prefer platform-supported codecs/containers for easy playback.
'''
