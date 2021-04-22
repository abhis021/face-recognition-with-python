import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    b, img = cam.read()
    if b:
        cv2.imshow("Face Detection", img)
        cv2.waitKey(1)
    else:
        print("The camera is not working.")
        break