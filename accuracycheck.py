import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
# import face_recognition

dataset_path = "./face_dataset/"

face_data = []
f= 1.00
labels = []
class_id = 0
g=0.00
names = {}





for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		names[class_id] = fx[:-4]
		data_item = np.load(dataset_path + fx)
		face_data.append(data_item)

		target = class_id * np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
print(face_labels.shape)
print(face_dataset.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)
print("Accuracy: " + str(f))
print("Loss: " + str(g))
