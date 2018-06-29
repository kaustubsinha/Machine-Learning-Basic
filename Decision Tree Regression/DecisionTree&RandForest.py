# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 23:51:20 2018

@author: Kaustub Sinha
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 23:49:57 2018

@author: Kaustub Sinha
"""

import pandas as pd
import numpy as np
rdline = pd.read_fwf("Auto_mpg.txt")
rdline.columns = ["mpg", "cylinders", "displacement","horsepower","weight","acceleration", "model year", "origin", "car name"]
for i in rdline:
    if rdline[i].dtype==object:
        rdline[i][rdline[i]=='?'] = rdline[i].mode()[0]
"""
To find Max Mpg
"""
rdline["car name"][rdline["mpg"] == rdline["mpg"].max()]

features = rdline.iloc[:,1:9].values
labels = rdline.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder
LB = LabelEncoder()
features[:,-1] = LB.fit_transform(features[:,-1])

from sklearn.model_selection import train_test_split as TTS
features_train, features_test,labels_train,labels_test=TTS(features,labels,test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor()
DTR.fit(features_train,labels_train)
ScoreDTR = DTR.score(features_test,labels_test)

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators =10, random_state=0)
RFR.fit(features_train,labels_train)
Pred = RFR.predict(features_test)
ScoreRFR = RFR.score(features_test,labels_test)

x = np.array([6,215,100,2630,22.2,80,3,12]).reshape(1,-1)
print(DTR.predict(x))
print(RFR.predict(x))