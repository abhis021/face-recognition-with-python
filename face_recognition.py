
#extratct the faces from the image and store into database
import cv2
import numpy as np
import pandas as pd
from os import listdir
from os.path import join ,isfile
face_classifier=cv2.CascadeClassifier(r"C:\Users\Abhishek Upadhyay\Desktop\image-data\haarcascade_frontalface_default.xml")
def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return None
    for (x,y,w,h) in faces:
        cropped_face=img[y:y+h,x:x+w]
        return cropped_face
    
cap=cv2.VideoCapture(0)        
count=0
while True:
    ret,frame=cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face=cv2.resize(face_extractor(frame),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        file_path=r"C:\Users\Abhishek Upadhyay\Desktop\image-data\user"+ str(count)+".jpg"
        cv2.imwrite(file_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("face_cropper",face)
    else:
        print("face not found")
    key = cv2.waitKey(5000) & 0xFF
    if key == 27 or count==100:
        break
cap.release()
cv2.destroyAllWindows()
print("successful")

#train the model on database image
import cv2
import numpy as np
from os import listdir
from os.path import join ,isfile
data_path='file_1'
only_files=[f for f in listdir(data_path) if isfile(join(data_path,f))]
training_data,labels=[],[]
for i ,files in enumerate (only_files):
    img_path=data_path +only_files[i]
    images=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    training_data.append(np.asarray(images,dtype=np.uint8))
    labels.append(i)
labels=np.asarray(labels,dtype=np.int32)
model=cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(training_data),np.asarray(labels))
print("complete")

#code to compare face with database image and recognised face 
import cv2
import numpy as np
import pandas as pd
from os import listdir
from os.path import join ,isfile
face_classifier=cv2.CascadeClassifier(r"C:\Users\Abhishek Upadhyay\Desktop\image-data\haarcascade_frontalface_default.xml")

def face_detector(img,size=0.5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return img,[]
    for (x,y,w,h) in faces:
        cropped_face=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize=(roi,(200,200))
    return img,roi
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    image,face=face_detector(frame)
    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result=model.predict(face)
        if result[1]<500:
            confidence=int(100*(1-(result[1])/300))
            display_string=str(confidence)+"%confidence"
        cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
        if confidence>75:
            cv2.putText(image,"unlocked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow("face_cropper",image)
        else:
            cv2.putText(image,"locked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow("face_cropper",image)
    except:
        cv2.putText(image,"face not found",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("face_cropper",image)
    key = cv2.waitKey(5000) & 0xFF
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()