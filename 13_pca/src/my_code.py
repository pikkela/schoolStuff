import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

inputfile='in.npy'
outputfile='out.npy'

data=np.load(inputfile)
# Teach  
pca = PCA()
pca.fit(data)

max_eigenvalue = pca.explained_variance_

min_eigenvalue = 0.1*max_eigenvalue.max()
newData = np.delete(max_eigenvalue, np.where(max_eigenvalue < min_eigenvalue))

N_packed = newData.shape[0]

del pca
pca = PCA(N_packed)
packed_data = pca.fit_transform(data)

np.save(outputfile, packed_data)
