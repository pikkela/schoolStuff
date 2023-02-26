import sys
import time
import numpy as np


X=np.load('teach_data.npy')
Y=np.load('teach_class.npy')

################################################
#Your code below this line



#Your code above this line
################################################

print('Compute real predictions')
real_X=np.load('data_in.npy')

print('real_X -', np.shape(real_X))
pred = model.predict(real_X)
print('pred -', np.shape(pred))
np.save('data_classified.npy', pred)

