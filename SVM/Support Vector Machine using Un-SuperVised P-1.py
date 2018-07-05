# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 13:34:00 2018

@author: Kaustub Sinha
"""

#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#
#dataset = pd.read_csv("Match_Making.csv")
#features = dataset.iloc[:, :-1].values
#labels = dataset.iloc[:, -1].values
#
## Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 0)
#
## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#features_train = sc.fit_transform(features_train)
#features_test = sc.transform(features_test)
#
## Fitting Kernel SVM to the Training set
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(features_train, labels_train)
#
## Predicting the Test set results
#labels_pred = classifier.predict(features_test)
#
## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(labels_test, labels_pred)
#
## Model Score
#score = classifier.score(features_test,labels_test)
#
## Visualising the Training set results
#from matplotlib.colors import ListedColormap
#features_set, labels_set = features_train, labels_train
#X1, X2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(labels_set)):
#    plt.scatter(features_set[labels_set == j, 0], features_set[labels_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('Kernel SVM (Training set)')
#plt.xlabel('Age')
#plt.ylabel('Estimated Salary')
#plt.legend()
#plt.show()
#
## Visualising the Test set results
#from matplotlib.colors import ListedColormap
#features_set, labels_set = features_test, labels_test
#X1, X2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(labels_set)):
#    plt.scatter(features_set[labels_set == j, 0], features_set[labels_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('Kernel SVM (Test set)')
#plt.xlabel('Age')
#plt.ylabel('Estimated Salary')
#plt.legend()
#plt.show()

'''
Code Challenge
'''
import numpy as np
import pandas as pd

dataset = pd.read_csv("breast_cancer.csv")
for i in dataset:
    dataset[i]=dataset[i].fillna(dataset[i].mode()[0])
    
features = dataset.iloc[:, 1:10].values
labels = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(features_train, labels_train)

labels_pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)

score = classifier.score(features_test,labels_test)

wf=np.array([6,2,5,3,2,7,9,2,4]).reshape(1,-1)
Pred_Girl = classifier.predict(wf)