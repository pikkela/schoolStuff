import pandas
pandas.options.mode.chained_assignment = None  # default='warn'
from scipy import stats
import numpy as np
from sklearn import preprocessing

inputfile='time_series.csv'
trainfile='train.csv'
testfile='test.csv'

data=pandas.read_csv(inputfile)
A = data.loc[(data['region'] == 'DE') & (data['attribute'] == 'generation_actual') & (data['variable'] == 'wind')]

A.drop(['region','attribute','variable'], axis=1, inplace=True)
A.dropna(inplace=True) #remove rows with NaN

#Remove outliers
z_threshold=4.0
z = np.abs(stats.zscore(A['data']))
A = A[z < z_threshold] #remove outliers

#normalize data column
A1 = A['data'].values.reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler()
A['data'] = min_max_scaler.fit_transform(A1)

#Split training set and test set
train_fraction=0.7

#print("Create training data set
traindata=A.sample(frac=train_fraction,random_state=200) #random state is a seed value

#Create test data set
testdata=A.drop(traindata.index)

#Save train data
traindata.to_csv(trainfile)

#Save test data
testdata.to_csv(testfile)

