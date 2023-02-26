import sys
import time
import pandas
from scipy import stats
import numpy as np
from sklearn import linear_model

from matplotlib import pyplot as plt
 
inputfile='measurements.csv'
m=0.75 #mass of the object

def readFile(x, y, length):

    #create linear regression model
    regr = linear_model.LinearRegression()
    #train the model using the training set
    regr.fit(x, y)

    # predict  values
    y_estimate = regr.predict(data_x)

    y_len = len(y_estimate)-1
    x_len = len(x)-1
    
    #calculate Ominaislämpökapasiteetti
    c =  y_estimate[y_len] / (m*x[0]-x[x_len])

    return float(c)

data=pandas.read_csv(inputfile) 
#drop nan:s
data.dropna(inplace=True)
#drop duplicates
data.drop_duplicates(inplace=True)

#Remove outliers
z_threshold=3.0
z = np.abs(stats.zscore(data['T']))
outlier=(z>=z_threshold)
data = data[z < z_threshold] #remove outliers
z = np.abs(stats.zscore(data['E']))
outlier=(z>=z_threshold)
data = data[z < z_threshold] #remove outliers

length = len(data)
data_x = data['T'].to_numpy().reshape(length,1)
data_y = data['E'].to_numpy().reshape(length,1)
c = readFile(data_x, data_y, length)
#Estimated specific heat capacity 
print(c)
