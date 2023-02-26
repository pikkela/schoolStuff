import sys
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

X=np.load('teach_data.npy')
Y=np.load('teach_class.npy')

################################################
#Your code below this line
train_fraction = 0.8

train_X, test_X, train_Y, test_Y =  train_test_split(X, Y, test_size=1-train_fraction)

model = svm.SVC()
model.fit(train_X, train_Y)

#Your code above this line
################################################

print('Compute real predictions')
real_X=np.load('data_in.npy')

print('real_X -', np.shape(real_X))
pred = model.predict(real_X)
print('pred -', np.shape(pred))
np.save('data_classified.npy', pred)

