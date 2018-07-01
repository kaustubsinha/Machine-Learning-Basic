# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 01:01:42 2018

@author: Kaustub Sinha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("deliveryfleet.csv")
features = data.iloc[:,1:].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans =KMeans(n_clusters=i, init = 'k-means++',random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("Numbers of Clusters")
plt.ylabel("WCSS")
plt.show()

#kmeans = KMeans(n_clusters=2, init = 'k-means++',random_state=0)
kmeans = KMeans(n_clusters=4, init = 'k-means++',random_state=0)
y_kmeans = kmeans.fit_predict(features)

plt.scatter(features[y_kmeans == 0, 0],features[y_kmeans == 0, 1], s= 100, c ='red',label = "Cluster1")
plt.scatter(features[y_kmeans == 1, 0],features[y_kmeans == 1, 1], s= 100, c ='Blue',label = "Cluster2")
plt.scatter(features[y_kmeans == 2, 0],features[y_kmeans == 2, 1], s= 100, c ='green',label = "Cluster3")
plt.scatter(features[y_kmeans == 3, 0],features[y_kmeans == 3, 1], s= 100, c ='brown',label = "Cluster4")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s =300, c ='yellow', label ='Centroid')

plt.title("Graph Distance Vs Speed")
plt.xlabel("Distance_Feature")
plt.ylabel("Speeding_Feature")
plt.legend()
plt.show()



mport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("tshirts.csv")
features = data.iloc[:,1:].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans =KMeans(n_clusters=i, init = 'k-means++',random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("Numbers of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=3, init = 'k-means++',random_state=0)
y_kmeans = kmeans.fit_predict(features)

plt.scatter(features[y_kmeans == 0, 0],features[y_kmeans == 0, 1], s= 100, c ='red',label = "Cluster1")
plt.scatter(features[y_kmeans == 1, 0],features[y_kmeans == 1, 1], s= 100, c ='Blue',label = "Cluster2")
plt.scatter(features[y_kmeans == 2, 0],features[y_kmeans == 2, 1], s= 100, c ='green',label = "Cluster3")
#plt.scatter(features[y_kmeans == 3, 0],features[y_kmeans == 3, 1], s= 100, c ='Blue',label = "Cluster4")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s =300, c ='yellow', label ='Centroid')

plt.title("Customer Height and Weight")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.legend()
plt.show()

