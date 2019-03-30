import cv2
import numpy as np
import matplotlib as plt
import time

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('E:\pythoncode\Tugas-Akhir\K15video_clip_1.AVI')
tic = time.time()

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    filter = cv2.fastNlMeansDenoising(frame, None, 11, 7, 21)
    edged_frame = cv2.Canny(filter, 100, 200)


    if ret:
        cv2.imshow("Echocardiography", frame)
        cv2.imshow("Canny Edge Detection", edged_frame)
        cv2.imshow("Filter", filter)
        toc = time.time()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Press X on keyboard to  exit
    if cv2.waitKey(50) & 0xFF == ord('x'):
        break
    print(toc - tic)

cap.release()
cv2.destroyAllWindows()