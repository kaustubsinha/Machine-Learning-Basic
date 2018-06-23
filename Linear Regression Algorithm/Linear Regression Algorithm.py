# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 10:42:12 2018

@author: Kaustub Sinha
"""
"""
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("Income_Data.csv")
features = dataset.iloc[:,:1].values
lables = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split as TT
features_train, features_test, labels_train, labels_test = TT(features,lables,test_size = 0.2 , random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train,labels_train)

label_pred = regressor.predict(features_test)

#Visulalizing the Training DataSet
plt.scatter(features_train, labels_train, color = 'red')
plt.plot(features_train,regressor.predict(features_train),color='blue')
plt.title("Income vs ML - Experience (Training set))")
plt.xlabel("ML-Experience")
plt.ylabel("Income")
plt.show()

#Visulalizing the Test DataSet
plt.scatter(features_test, labels_test, color = 'red')
plt.plot(features_train,regressor.predict(features_train),color='blue')
plt.title("Income vs ML - Experience (Test set))")
plt.xlabel("ML-Experience")
plt.ylabel("Income")
plt.show()
"""

"""
Q1:
    
You will implement linear regression to predict the profits for a food chain 
company.

Case: Suppose you are the CEO of a restaurant franchise and are considering 
different cities for opening a new outlet. The chain already has food-trucks 
in various cities and you have data for profits and populations from the cities You would like to use this data to help you select which city to expand to next. 

Foodtruck.csv contains the dataset for our linear regression problem. The 
first column is the population of a city and the second column is the profit 
of a food truck in that city. A negative value for profit indicates a loss.

Perform Simple Linear regression to predict the profit based on the population 
observed and visualize the result.

Based on the above trained results, what 
will be your estimated profit, if you set up your outlet in Jaipur? (Current 
population in Jaipur is 3.073 million)
"""
"""
Perform Simple Linear regression to predict the profit based on the population 
observed and visualize the result.
"""
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("Foodtruck.csv")
features_Pop = dataset.iloc[:,:1].values
lables_Prof = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split as TT
features_train, features_test, labels_train, labels_test = TT(features_Pop,lables_Prof,test_size = 0.2 , random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train,labels_train)
label_pred = regressor.predict(features_test)
label_pred_Jaipur = regressor.predict(3.073)
print(label_pred)
print(" ")
print("Prediction For Jaipur",label_pred_Jaipur)

#Visulalizing the Training DataSet
plt.scatter(features_train, labels_train, color = 'red')
plt.plot(features_train,regressor.predict(features_train),color='blue')
plt.title("Population Vs Profit(Training set))")
plt.xlabel("Population")
plt.ylabel("Profit")
plt.show()

#Visulalizing the Test DataSet
plt.scatter(features_train, labels_train, color = 'red')
plt.plot(features_train,regressor.predict(features_train),color='blue')
plt.title("Population Vs Profit(Training set))")
plt.xlabel("Population")
plt.ylabel("Profit")
plt.show()







"""
Q:2 
Import Bahubali2vsDangal.csv file.

It contains Data of Day wise collections of the movies Bahubali 2 and Dangal 
(in crores) for the first 9 days. Now, you have to write a python code to 
predict which movie would collect more on the 10th day.
"""

import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("Bahubali2_vs_Dangal.csv")
features = dataset.iloc[:,:1].values
lables_BahuBali = dataset.iloc[:,-2].values
lables_Dangal = dataset.iloc[:,-1].values

#for Test Train for Bahubali
from sklearn.model_selection import train_test_split as TT
features_trainB, features_testB, labels_trainB, labels_testB = TT(features,lables_BahuBali,test_size = 0.2 , random_state=0)

#for Test Train Dangle
from sklearn.model_selection import train_test_split as TT
features_trainD, features_testD, labels_trainD, labels_testD = TT(features,lables_Dangal,test_size = 0.2 , random_state=0)

#for Test Train for Bahubali
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_trainB,labels_trainB)
label_predB = regressor.predict(10)
print('for BahuBali :', label_predB)

#for Test Train for Dangle
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_trainD,labels_trainD)
label_predD = regressor.predict(10)
print('for Dangle :', label_predD)

plt.scatter(features_testB, labels_testB, color = 'red')
plt.plot(features_trainB,regressor.predict(features_trainB),color='blue')
plt.title("Bahubali Vs Days (Test set))")
plt.xlabel("Days")
plt.ylabel("Income")
plt.show()

plt.scatter(features_testD, labels_testD, color = 'red')
plt.plot(features_trainD,regressor.predict(features_trainD),color='blue')
plt.title("Dangle Vs Days (Test set))")
plt.xlabel("Days")
plt.ylabel("Income")
plt.show()