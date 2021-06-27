import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('face_dataset.csv')

# X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3) # 70% training and 30% test

# #create KNN Classifier
# knn = KNeighborsClassifier(n_neighbors=7)
# #train the model using training set
# knn.fit(X_train,y_train)

# y_pred = knn.predict(X_test)

# fit the model
r = dataset.fit_generator(
  dataset,
  validation_data=dataset,
  epochs=5,
  steps_per_epoch=len(dataset),
  validation_steps=len(dataset)
)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
