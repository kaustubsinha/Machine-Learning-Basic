# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:43:41 2018

@author: Kaustub Sinha
"""

"""
Sample First

"""


#import numpy as np
#import pandas as pd
#dataset  = pd.read_csv("Salary_Classification.csv")
#features = dataset.iloc[:,:-1].values
#labels = dataset.iloc[:,-1].values
#
#from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#LB = LabelEncoder()
#features[:,0] = LB.fit_transform(features[:,0])
#
#onehotencoder= OneHotEncoder(categorical_features=[0])
#features = onehotencoder.fit_transform(features).toarray()
#
#features = features[:,1:]
#
#from sklearn.model_selection import train_test_split as TTS
#features_train, features_test,labels_train,labels_test=TTS(features,labels,test_size=0.2,random_state=0)
#
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#features_train = sc.fit_transform(features_train)
#features_test = sc.fit_transform(features_test)
#
#from sklearn.linear_model import LinearRegression
#LR = LinearRegression()
#LR.fit(features_train,labels_train)
#
#Pred = LR.predict(features_test)
#Score = LR.score(features_train,labels_train)
#
#import statsmodels.formula.api as sm
#features = np.append(arr = np.ones((30,1)).astype(int), values= features, axis= 1)
#
#features_opt=features[:,[0,1,2,3,4,5]]
#LR_OLS = sm.OLS(endog=labels, exog=features_opt).fit()
#LR_OLS.summary()
#
#features_opt=features[:,[0,1,3,4,5]]
#LR_OLS = sm.OLS(endog=labels, exog=features_opt).fit()
#LR_OLS.summary()
#
#features_opt=features[:,[0,1,3,5]]
#LR_OLS = sm.OLS(endog=labels, exog=features_opt).fit()
#LR_OLS.summary()
#
#features_opt=features[:,[0,3,5]]
#LR_OLS = sm.OLS(endog=labels, exog=features_opt).fit()
#LR_OLS.summary()

"""
Sample Second




"""
#import numpy as np
#import pandas as pd
#dataset  = pd.read_csv("iq_size.csv")
#features = dataset.iloc[:,:-1].values
#labels = dataset.iloc[:,-1].values
#
#from sklearn.model_selection import train_test_split as TTS
#features_train, features_test,labels_train,labels_test=TTS(features,labels,test_size=0.2,random_state=0)
#
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#features_train = sc.fit_transform(features_train)
#features_test = sc.fit_transform(features_test)
#
#from sklearn.linear_model import LinearRegression
#LR= LinearRegression()
#LR.fit(features_train,labels_train)
#
#Pred = LR.predict(features_test)
#Score = LR.score(features_train,labels_train)
#
#import statsmodels.formula.api as sm
#features = np.append(arr = np.ones((38,1)).astype(int),values=features, axis=1)
#
#features_opt=features[:,[0,1,2]]
#LR_OLS = sm.OLS(endog=labels, exog=features_opt).fit()
#LR_OLS.summary()
#
#features_opt=features[:,[1,2]]
#LR_OLS = sm.OLS(endog=labels, exog=features_opt).fit()
#LR_OLS.summary()

"""
Sample Third




"""
#import numpy as np
#import pandas as pd
#dataset  = pd.read_csv("stats_females.csv")
#features = dataset.iloc[:,:-1].values
#labels = dataset.iloc[:,-1].values
#
#
#from sklearn.model_selection import train_test_split as TTS
#features_train, features_test,labels_train,labels_test=TTS(features,labels,test_size=0.2,random_state=0)
#
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#features_train = sc.fit_transform(features_train)
#features_test = sc.fit_transform(features_test)
#
#from sklearn.linear_model import LinearRegression
#LR= LinearRegression()
#LR.fit(features_train,labels_train)
#
#Pred = LR.predict(features_test)
#Score = LR.score(features_train,labels_train)
#
#import statsmodels.formula.api as sm
#features = np.append(arr = np.ones((214,1)).astype(int), values= features, axis= 1)
#
#features_opt=features[:,[0,1,2]]
#LR_OLS = sm.OLS(endog=labels, exog=features_opt).fit()
#LR_OLS.summary()
