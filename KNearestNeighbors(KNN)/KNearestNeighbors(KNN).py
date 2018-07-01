# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 00:11:31 2018

@author: Kaustub Sinha
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 14:37:37 2018

@author: Kaustub Sinha
"""

import numpy as np
import pandas as pd

data = pd.read_csv("mushrooms.csv")
features = data.iloc[:,1:-1].values
labels = data.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder
LB = LabelEncoder()
for i in range(0,21):
    features[:,i] = LB.fit_transform(features[:,i])
features1 = pd.DataFrame(features)
#onehotencoder = OneHotEncoder(categorical_features = [5,-2,-1])
#features = onehotencoder.fit_transform(features).toarray()

from sklearn.model_selection import train_test_split as tts
features_train, features_test,labels_train,labels_test = tts(features,labels,test_size = 0.25, random_state =0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,p=2)
classifier.fit(features_train,labels_train)

Pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,Pred)

Score = classifier.score(features_test,labels_test)

#Perform Classification on the given dataset to predict if the mushroom is 
#edible or poisonous w.r.t. it’s different attributes.
#5	2	9	1	0	1	0	0	4	0	2	2	2	7	7	0	2	1	4	3	2
features1.columns = ["cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing	","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","	stalk-surface-below-ring","	stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color","ring-number","	ring-type","spore-print-color","population","habitat"]
x = np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,2]).reshape(1,-1)
EatOrNot = classifier.predict(x)
print(EatOrNot)
