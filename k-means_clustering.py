# 25 April 2025
# K- Means Clustering
'''
Depedent variable in not present in clustering
clustering is unsupervised learning
data is unstructured in clustering

Key point for k-means:
    
centroid concept
new centroid calculation
Euclidiean distance
elbow method


Datasets:
    Spending score is discrete value (from 1-100 only)
    Gender & Age - Irrelvalant attribute (remove them)

'''
    
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\GauravKunal\Desktop\DS\Machine Learning\#3 Clustering\Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values # annual income & spending score only
X

from sklearn.cluster import KMeans

# Elbow graph on Y-axis we have wcss
wcss = []


for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=0)
    kmeans.fit(X)
    '''n_init'auto' or int, default='auto'
Number of times the k-means algorithm is run with different centroid seeds. The final results is the best output of n_init consecutive runs in terms of inertia. Several runs are recommended for sparse high-dimensional problems (see kmeans_sparse_high_dim).

When n_init='auto', the number of runs depends on the value of init: 10 if using init='random' or init is a callable; 1 if using init='k-means++' or init is an array-like.

Added in version 1.2: Added 'auto' option for n_init.

Changed in version 1.4: Default value for n_init changed to 'auto'.'''
    wcss.append(kmeans.inertia_)
    
# Elbow Graph
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel("WCSS")
plt.show()


# According to the graph we have to make 5 clusters

kmeans = KMeans(n_clusters= 5 , init="k-means++", random_state=0)
y_kmeans = kmeans.fit_predict(X)


dataset['y_kmeans'] = y_kmeans

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()



plt.scatter(x, y, kwargs)







































