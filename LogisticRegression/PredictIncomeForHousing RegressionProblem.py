# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 12:17:50 2018

@author: Kaustub Sinha
"""
import numpy as np
import pandas as pd 
dataset  = pd.read_csv("income.csv")

for i in dataset:
    if dataset[i].dtype==object:
        dataset[i][dataset[i]=='?'] = dataset[i].mode()[0]
features = dataset.iloc[:,:-1].values
labels = dataset.iloc[:,-1].values
features1 = pd.DataFrame(features)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LB = LabelEncoder()

for i in [1,3,5,6,7,8,9,13]:
    features[:,i] = LB.fit_transform(features[:,i])
labels = LB.fit_transform(labels)

from sklearn.model_selection import train_test_split as tts
features_train, features_test, labels_train,labels_test = tts(features,labels,test_size=0.2, random_state=0)

onehotencoder= OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
features_train = onehotencoder.fit_transform(features_train).toarray()
features_test = onehotencoder.transform(features_test).toarray()

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

PredIncome = onehotencoder.transform(np.array([38,3,89814,11,9,2,4,0,4,1,0,0,50,38]).reshape(1,-1)).toarray()
print(classifier.predict(PredIncome))