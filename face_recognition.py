import numpy as np
import cv2
import os

########## KNN CODE ########
def distance (v1, v2):
    #Eucledian
    return np.sqrt(((v1-v2)**2).sum())

def knn (train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        #Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        #compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    #sort based on distance and get top k
    dk = sorted(dist, key = lambda x:x[0])[:k]
    #retrieve only the labels
    labels = np.array(dk)[:, -1]
    #Get frequencies of each label
    output=np.unique(labels, return_counts=True)
    #Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]
############################################################

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("C:\\Users\\Abhishek Upadhyay\\Documents\\GitHub\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt.xml")

dataset_path = "./face_dataset"

face_data = []
labels = []
class_id = 0
names = {}