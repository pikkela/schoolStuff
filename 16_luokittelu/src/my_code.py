import pandas 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm

filename1='grading.csv'
train_fraction = 0.8
Y_column='Passed'

#Load data
data=pandas.read_csv(filename1)
print("Read data shape = "+str(data.shape))
print(data)

#################################################
#Your code here
#remove rows with zero in it
data = data[data['Assignment A'] !=0]
data = data[data['Assignment B'] !=0]
data = data[data['Assignment C'] !=0]

#split data by 'Passed' and test score
train_fraction = 0.8
X = data.drop(['Name', 'Passed'], axis=1)
Y = data['Passed']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1-train_fraction)
#
#Create a classifier that classifies students
classifier = svm.SVC()
classifier.fit(X_train,y_train)

#
#################################################

#Load real data
filename2='assignments.csv'
data=pandas.read_csv(filename2)
names=data['Name']

#Remove name column
for col in ["Name"]:
    print("Remove "+col)
    data.drop(col, axis=1, inplace=True)
print()

print("Read data shape = "+str(data.shape))
print(data)
predY=classifier.predict(data)

#Create dataframe from numpy data
df = pandas.DataFrame({'Name': names, 'Passed': predY})
print(df)
df.to_csv('prediction.csv', index=False)

