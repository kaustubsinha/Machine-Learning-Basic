# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:15:57 2018

@author: Kaustub Sinha
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values
# Splitting the dataset into the Training set and Test
set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()
- 1, stop = X_set[:, 0].max() + 1, step = 0.01),
 np.arange(start = X_set[:, 1].min()
- 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2,
classifier.predict(np.array([X1.ravel(),
X2.ravel()]).T).reshape(X1.shape),
 alpha = 0.75, cmap = ListedColormap(('red',
'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
 plt.scatter(X_set[y_set == j, 0], X_set[y_set == j,
1],
 c = ListedColormap(('red', 'green',
'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()
- 1, stop = X_set[:, 0].max() + 1, step = 0.01),
 np.arange(start = X_set[:, 1].min()
- 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2,
classifier.predict(np.array([X1.ravel(),
X2.ravel()]).T).reshape(X1.shape),
 alpha = 0.75, cmap = ListedColormap(('red',
'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
 plt.scatter(X_set[y_set == j, 0], X_set[y_set == j,
1],
 c = ListedColormap(('red', 'green',
'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

'''
Code Challenge 1

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('crime_data.csv')
X = dataset.iloc[:, [1,2,4]].values

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans =KMeans(n_clusters=i, init = 'k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("Numbers of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=3, init = 'k-means++',random_state=0)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0],X[y_kmeans == 0, 1], s= 100, c ='red',label = "Cluster1")
plt.scatter(X[y_kmeans == 1, 0],X[y_kmeans == 1, 1], s= 100, c ='Blue',label = "Cluster2")
plt.scatter(X[y_kmeans == 2, 0],X[y_kmeans == 2, 1], s= 100, c ='green',label = "Cluster3")
#plt.scatter(features[y_kmeans == 3, 0],features[y_kmeans == 3, 1], s= 100, c ='Blue',label = "Cluster4")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s =300, c ='yellow', label ='Centroid')

plt.title("Crime Chart")
plt.xlabel("Y-Axis")
plt.ylabel("X-Axis")
plt.legend()
plt.show()

new_df = pd.DataFrame()
new_df["City"] = dataset["State"]
new_df["Crime_Groups"] = y_kmeans
new_df["City"][new_df["Crime_Groups"]==1]


'''
Code Challenge 2

'''

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()
iris=iris.data

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
iris = pca.fit_transform(iris)
explained_variance = pca.explained_variance_ratio_

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans =KMeans(n_clusters=i, init = 'k-means++',random_state=0)
    kmeans.fit(iris)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("Numbers of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=3, init = 'k-means++',random_state=0)
y_kmeans = kmeans.fit_predict(iris)

plt.scatter(iris[y_kmeans == 0, 0],iris[y_kmeans == 0, 1], s= 100, c ='red',label = "Cluster1")
plt.scatter(iris[y_kmeans == 1, 0],iris[y_kmeans == 1, 1], s= 100, c ='Blue',label = "Cluster2")
plt.scatter(iris[y_kmeans == 2, 0],iris[y_kmeans == 2, 1], s= 100, c ='green',label = "Cluster3")
#plt.scatter(features[y_kmeans == 3, 0],features[y_kmeans == 3, 1], s= 100, c ='Blue',label = "Cluster4")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s =300, c ='yellow', label ='Centroid')

plt.title("Dimension Reduction For Flower Data")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.legend()
plt.show()