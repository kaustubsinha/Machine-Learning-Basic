# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 14:03:46 2018

@author: Kaustub Sinha
"""
import pandas as pd
from apyori import apriori
data =  pd.read_csv("Market_Basket_Optimisation.csv", header = None) 
transaction=[]
for i in range(0,7501):
    transaction.append([str(data.values[i,j])for j in range(20)])

rule = apriori(transaction,min_support=0.003,max_support=0.2,min_lift=3,min_length=2)
results = list(rule)