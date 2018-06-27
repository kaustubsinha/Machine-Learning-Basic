# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 22:42:28 2018

@author: Kaustub Sinha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("bluegills.csv")
features = dataset.iloc[:,:-1].values
lables = dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(features,lables)
ScoreLR = LR.score(features,lables)

plt.scatter(features,lables,color ='red')
plt.plot(features,LR.predict(features),color='blue')
plt.title("Graph for Bluegills(Linear Regression")
plt.xlabel("Age")
plt.ylabel("Length")
plt.show()

from sklearn.preprocessing import PolynomialFeatures

poln_object = PolynomialFeatures(degree = 4)
features_poln = poln_object.fit_transform(features)

LR2 = LinearRegression()
LR2.fit(features_poln,lables)

Pred = LR2.predict(poln_object.fit_transform(5))
ScoreLR2 = LR2.score(features_poln,lables)

plt.scatter(features,lables,color ='red')
plt.plot(features,LR2.predict(poln_object.fit_transform(features)),color='blue')
plt.title("Graph for Bluegills(Ploy Linear Regression")
plt.xlabel("Age")
plt.ylabel("Length")
plt.show()

features_grid = np.arange(min(features),max(features),0.1)
features_grid = features_grid.reshape(-1,1)
plt.scatter(features,lables,color ='red')
plt.plot(features_grid,LR2.predict(poln_object.transform(features_grid)),color='blue')
plt.title("Graph for Bluegills(Ploy Linear Regression")
plt.xlabel("Age")
plt.ylabel("Length")
plt.show()