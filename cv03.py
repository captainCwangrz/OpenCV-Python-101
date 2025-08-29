"""
Unit 3: Webcam Preview & FPS
- Overview: Open webcam, show live frames, and overlay FPS (running average).
- Inputs: Default camera index 0.
- Usage: Press 'q' in the preview window to exit.
"""

import cv2, time, collections

# --- Camera setup ---
# Index 0 selects the default camera. A filename like 'video.mp4' also works.
cap = cv2.VideoCapture(0)

'''
These settings may or may not work depending on your camera and OpenCV backend

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS,          60)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps_reported = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera reports: {w}x{h} @ {fps_reported} FPS (may be unreliable)")
'''

# --- FPS running average over last 30 frames ---
times = collections.deque(maxlen=30)
start = time.time()
while True:
  ret, frame = cap.read() # ret is whether the frame was grabbed
  if not ret:
    print("Failed to grab frame")
    break
  
  # Calculate FPS using running average over last 30 frames
  # Notes: typical webcams run ~30 FPS @ 1080p; exposure, camera caps,
  # backend, USB bandwidth, and drivers all affect measured FPS.
  now = time.time()
  times.append(now - start)
  start = now
  fps = 1 / (sum(times) / len(times))
  # Overlay FPS text: (org), font, scale, color (BGR), thickness
  cv2.putText(frame, f'FPS: {fps:.1f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

  cv2.imshow("Webcam", frame)

  # Wait ~1 ms to process GUI events; quit on 'q'
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()

'''
Unit 3 Summary
Main functions:
 - `cv2.VideoCapture(0)` — open default camera
 - `cap.read()` — grab frames in a loop
 - `cv2.putText()` — annotate frames
 - `cv2.imshow()` / `cv2.waitKey()` — display and handle UI
 - `cap.release()` / `cv2.destroyAllWindows()` — cleanup

Key ideas:
 - FPS via running average of frame intervals for stability.
 - Camera-reported properties (fps, width/height) can be unreliable.

Tips:
 - Tune exposure/auto-exposure; it impacts FPS and motion blur.
 - On Linux, `v4l2-ctl` helps list/set camera capabilities.
 - Consider threading or async capture for CPU-bound pipelines.
'''
