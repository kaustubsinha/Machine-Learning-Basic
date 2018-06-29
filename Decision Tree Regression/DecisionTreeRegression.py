# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 23:49:19 2018

@author: Kaustub Sinha
"""

#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#dataset = pd.read_csv("Position_Salaries.csv")
#features = dataset.iloc[:,1:2].values
#labels = dataset.iloc[:,2].values
#
#from sklearn.tree import DecisionTreeRegressor
#regressor = DecisionTreeRegressor(random_state=0)
#regressor.fit(features,labels)
#
#label_pred = regressor.predict(6.5)
#
#features_grid = np.arange(min(features),max(features),0.01)
#features_grid = features_grid.reshape(-1,1)
#plt.scatter(features,labels,color="red")
#plt.plot(features_grid,regressor.predict(features_grid), color="blue")
#plt.title("Grapth of Position vs Age")
#plt.xlabel("Position Order")
#plt.ylabel("Age")
#plt.show()

"""
Example 2
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("PastHires.csv")
features = dataset.iloc[:,:-1].values
labels = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
LB = LabelEncoder()
for i in [1,3,4,5]:
    features[:,i] = LB.fit_transform(features[:,i])
    
labels=LB.fit_transform(labels)
features1 = pd.DataFrame(features)

onehotencoder= OneHotEncoder(categorical_features=[0])
features = onehotencoder.fit_transform(features).toarray()

from sklearn.model_selection import train_test_split as tts
features_train,features_test,labels_train,labels_test = tts(features,labels,test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(features_train,labels_train)

Pred = regressor.predict(features_train)

Score = regressor.score(features_train,labels_train)

#Predict employment of a currently employed 10-year veteran, previous 
#employers 4, went to top-tire school, having Bachelor's Degree without 
#Internship.
#x = onehotencoder.transform(np.array([10,1,4,0,1,0]).reshape(1,-1)).toarray()
x = onehotencoder.transform(np.array([10, 1, 4, 0, 1, 0]).reshape(1,-1))
Pred1 = regressor.predict(x)

#Predict employment of an unemployed 10-year veteran, ,previous employers 4, 
#didn't went to any top-tire school, having Master's Degree with Internship.
y = onehotencoder.transform(np.array([7, 1, 1, 2, 1, 0]).reshape(1,-1)).toarray()
Pred2 = regressor.predict(y)
