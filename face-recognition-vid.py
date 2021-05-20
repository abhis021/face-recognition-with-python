import face_recognition
import argparse
import cv2
import pickle
import imutils, camera_access, predict_faces

ap=argparse.ArgumentParser()
ap.add_argument("-y", "--display", type = int, default=1, help= "whether or not to display output frame to screen")

args = vars(ap.parse_args())

print("Starting Video")

camera_access().start()

while True:
    frame = camera_access()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rgb = imutils.resize(frame, width = 750)
    r = frame.shape[1] / float(rgb.shape[1])

    boxes = face_recognition.face_locations(rgb, model = 'hog')
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []

    for encoding in encodings:
        encoding = [encoding.tolist()]
        name = predict_faces.main(encoding)
        names.append(name)
    for ((top, right, bottom, left), name) in zip(boxes, names):
        top = int(top*r)
        right = int(right*r)
        bottom = int(bottom*r)
        left = int(left*r)

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        y = top - 15 if top - 15 > 15 else top +15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
    
cv2.destroyAllWindows()