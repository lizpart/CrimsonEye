# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Reading the data
data = pd.read_csv('mnist_train_small.csv')

#Data preprocessing
data.drop(['City', 'State'], axis = 1, inplace = True)
data.fillna(data.mean(), inplace = True)

# Elbow method to determine the number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# K-means clustering
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
pred_y = kmeans.fit_predict(data)

# Visualization of clusters
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=pred_y)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.title('Crime Rate Clusters')
plt.xlabel('Murder Rate')
plt.ylabel('Assault Rate')
plt.show()


import pickle

# assume that the k-means model is stored in the variable kmeans_model
with open('crime_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)