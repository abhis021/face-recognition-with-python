import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = "./face_dataset/"

file_name = input("Enter the name of the person: ")

while True:
    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame.cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        continue
    k = 1

    faces= sorted(faces, key = lambda x:[2]*[3], reverse = True)
    skip +=1

    for face in faces[:1]:
        x, y, w, h = face
        offset = 5
        face_offset = frame[y-offset.y+h+offset,x-offset:x+w+offset]
        face_selection = cv2. resize(face_offset, (100, 100))

        if skip % 10 ==0:
            face_data.append(face_selection)
            print(len(face_data))
        
        cv2.imshow(str(k), face_selection)
        


        # this doesn't work, need to make this work and few other modules 
        # so what does this py does..??
        # this program will access face data and recognize
        #im following this tutorial

