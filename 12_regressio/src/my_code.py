import sys
import time
import pandas
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

inputfile='mittaus.csv'
modeldegree=2

data=pandas.read_csv(inputfile)
dataa = data.values.reshape((40, 1))
#convert to matrix
data_x=data['x'].values.reshape((len(data['x']-1), 1))
data_y=data['y'].values.reshape((len(data['y']-1), 1))

poly_reg=PolynomialFeatures(degree=modeldegree)
X_poly=poly_reg.fit_transform(data_x)

#fit data to linear model
linreg2=linear_model.LinearRegression()
linreg2.fit(X_poly,data_y)

y_estimate2=linreg2.predict(poly_reg.fit_transform(data_x))
############## Notes for me ######################## 
# fig=plt.figure(figsize=(16, 8))
# plt.plot(data_x, y_estimate2)
# #plt.plot(X_poly, y_estimate2,color='blue')
# #plt.plot(x_estiamte, data_y,color='k')
# plt.show()
####################################################

#fit data to second degree polynomial
fitted = np.polyfit(data_x.flatten(),y_estimate2.flatten(),2)

def secDeg(a, b,c):
    #
    # y=ax^+bx+c
    # -b+-sqr(b^2-4*(a^c))/2*a   
    #print(((-b)+(np.sqrt((b**2)-4*a*c)))/(2*a))
    return( (-b)-(np.sqrt((b**2)-4*a*c)))/(2*a)
    
print(secDeg(fitted[0], fitted[1],fitted[2]))

