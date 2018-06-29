# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 20:40:35 2018

@author: Kaustub Sinha
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
dataset  = pd.read_csv("affairs.csv")
features =  dataset.iloc[:,:-1].values
features_opt = dataset.iloc[:,:-1].values
labels = dataset.iloc[:,-1].values

#import statsmodels.formula.api as sm
#features = np.append(arr = np.ones((6366,1)).astype(int),values = features, axis=1)
#
#features_opt = features[:,[0,1,2,3,4,5,6,7,8]]
#regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
#regressor_OLS.summary()
#
#features_opt = features[:,[1,2,5,6,7,8]]
#regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
#regressor_OLS.summary()

from sklearn.model_selection import train_test_split as tts
features_train,features_test,labels_train,labels_test = tts(features_opt,labels,test_size=0.2, random_state=0)

from sklearn.preprocessing import OneHotEncoder
#onehotencoder = OneHotEncoder(categorical_features=[4,5])
onehotencoder = OneHotEncoder(categorical_features=[6,7])
features_train = onehotencoder.fit_transform(features_train).toarray()
features_test=onehotencoder.transform(features_test).toarray()



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(features_train,labels_train)

labels_Pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_Pred)

Score = classifier.score(features_test,labels_test)
#PredGirl = sc.transform(onehotencoder.transform(np.array([3,25,1,16,4,2]).reshape(1,-1)).toarray())
PredGirl = sc.transform(onehotencoder.transform(np.array([3,25,3,1,4,16,3,2]).reshape(1,-1)).toarray())
print(classifier.predict(PredGirl))
