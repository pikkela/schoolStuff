import sys
import time
import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

reduced_N=32

########################################################
#Write your code here
def setRange(x):
    range = (x/128)-1
    return range

#Set value range -1..1
train_X_range = setRange(train_X)
test_X_range = setRange(test_X)

#Convert figures to vectors
def convertToVector(data):
    nsamples, nx, ny = data.shape
    d2_train_dataset = data.reshape((nsamples,nx*ny))
    return d2_train_dataset

train_X_vector = convertToVector(train_X_range)
test_X_vector = convertToVector(test_X_range)
#Compute reduced PCA
def pcaTrain(data):
    pca = PCA(reduced_N)
    pca.fit(train_X_vector)
    trained = pca.transform(data)
    del pca
    return trained

train_X_packed = pcaTrain(train_X_vector)#reduced dimension
test_X_packed = pcaTrain(test_X_vector) #reduced dimension

# #End of your code
# ########################################################
# #Do not modify lines below this point!




#Save packed data
print('Save packed data')
np.save('packed_train.npy', train_X_packed)
np.save('packed_test.npy', test_X_packed)

if len(sys.argv)==1:
    #Test quality
    print('Train model')
    model = KNeighborsClassifier(n_neighbors = 11)
    model.fit(train_X_packed, train_Y)

    print('Compute predictions')
    pred = model.predict(test_X_packed)
    acc = accuracy_score(test_Y, pred)

    print('Accuracy =',acc)
