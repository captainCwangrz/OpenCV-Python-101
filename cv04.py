import cv2

cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h), isColor=False)
while True:
  ret, frame = cap.read()
  if not ret:
    print("Fialed to grab frame")
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
Unit 3
cv2.VideoWriter to save video to file
  - fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec
  - out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))
  - out.write(frame) to write each frame
  - out.release() when done
fps is important for correct video timing
fps parameter dictates video playback speed
  - if fps is too high, video plays too fast
  - if fps is too low, video plays too slow
So knowing the actual fps of your input is important
But specifics are beyong the scope of this
'''