# -*- coding: utf-8 -*-


 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values
 
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init=10, random_state= 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title("metodo del codo")
plt.xlabel("numero de clusters")
plt.ylabel("wcss(k)")
plt.show()


kmeans = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init=10, random_state= 0)

y_means = kmeans.fit_predict(X)