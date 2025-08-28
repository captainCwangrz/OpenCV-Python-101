import cv2, time, collections

cap = cv2.VideoCapture(0) # 0 is usually the built-in webcam, something like video.mp4 also works

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

times = collections.deque(maxlen=30)
start = time.time()
while True:
  ret, frame = cap.read() # ret is whether the frame was grabbed
  if not ret:
    print("Failed to grab frame")
    break
  
  # Calculate and display FPS using running average over last 30 frames
  # Seems like webcam fps is usually around 30 @ 1080p
  # Exposure, camera cap, backend, usb bandwidth, driver all affect fps
  now = time.time()
  times.append(now - start)
  start = now
  fps = 1 / (sum(times) / len(times))
  cv2.putText(frame, f'FPS: {fps:.1f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

  cv2.imshow("Webcam", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to quit, 1 ms delay is minimum for imshow to work
    break
cap.release()
cv2.destroyAllWindows()

'''
Unit 3 Summary
cv2.VideoCapture(0) to capture from webcam
cap.read() to read frames in a loop
cap.release() to release the camera

For serious applications:
 - Must know your camera's supported resolutions, fps, exposure, etc.
 - Settings matter a lot
 - Use v4l2-ctl on Linux to get/set camera settings?
 - Consider threading for performance (depends on application real time vs motion analysis)
'''
