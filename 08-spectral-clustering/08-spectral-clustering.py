#!/usr/bin/env python
# coding: utf-8

# ### DASC521: Intro to ML
# ### Homework 8: Spectral Clustering
# ### Gamze Keçibaş 60211  
# ---
# ### CONTENT
# - **Step 01.** Import libraries    
# - **Step 02.** Import dataset & Data preparation  
# - **Step 03.** Define learning functions & clustering  
# - **Step 04.** Plot clusters    

# - **Step 01.** Import libraries  

# In[1]:


import math
import numpy as np
import scipy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.spatial as spa
from scipy.sparse import csgraph

np.random.seed(421)


# - **Step 02.** Import dataset & Data preparation    

# In[2]:


dataset = np.genfromtxt('hw08_data_set.csv',delimiter=',')
print(dataset.shape)
X = dataset[:,0]
Y = dataset[:,1]

K= 5                                # number of cluster
N = [50, 50, 50, 50, 100]                              # number of samples in classes except C5

mu1 = [2.5, 2.5]                    # mean array of classes
mu2 = [-2.5, 2.5]
mu3 = [-2.5, -2.5]
mu4 = [2.5, -2.5]
mu5 = [0.0, 0.0]

sigma1 = [[0.8, -0.6] , [-0.6, 0.8]]    # covariance matrix of classes
sigma2 = [[0.8, 0.6] , [0.6, 0.8]]
sigma3 = [[0.8, -0.6],[-0.6, 0.8]]
sigma4 = [[0.8, 0.6] , [0.6, 0.8]]
sigma5 = [[1.6, 0.0],[0.0, 1.6]]

plt.figure(figsize= (8,8))
plt.scatter(X,Y,color='black')
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.title('Provided Dataset')
plt.show()


# - **Step 03.** Define learning functions & clustering  

# In[3]:


dist_threshold = 1.25
euc_dist = np.zeros([len(dataset[:,0]),len(dataset[:,0])])
plt.figure(figsize= (8,8))
for i in range(len(dataset[:,0])):
    for j in range(len(dataset[:,0])):
        euc_dist[i,j]= spa.distance.euclidean(dataset[i,:], dataset[j,:])
        if euc_dist[i,j] < dist_threshold:
            plt.plot([dataset[i,0], dataset[j,0]],[dataset[i,1], dataset[j,1]], color= 'gray')
plt.scatter(X,Y,color='black',zorder=5)
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.title('Euclidean Distances where delta = 1.25')
plt.show()


# In[4]:


new_centroids= [dataset[28],dataset[142],dataset[203],dataset[270],dataset[276]]
def laplacian(A):
    """Computes the symetric normalized laplacian.
    L = D^{-1/2} A D{-1/2}
    """
    D = np.zeros(A.shape)
    w = np.sum(A, axis=0)
    D.flat[::len(w) + 1] = w ** (-0.5)  # set the diag of D to w
    return D.dot(A).dot(D)


def k_means(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1231)
    return kmeans.fit(X).labels_


def spectral_clustering(affinity, n_clusters, cluster_method=k_means):
    L = laplacian(affinity)
    eig_val, eig_vect = scipy.sparse.linalg.eigs(L, n_clusters)
    X = eig_vect.real
    rows_norm = np.linalg.norm(X, axis=1, ord=2)
    Y = (X.T / rows_norm).T
    labels = cluster_method(Y, n_clusters)
    return labels

def plot_current_state(new_centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"])
    plt.figure(figsize=(12,12))
    if memberships is None:
        plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10, color = cluster_colors[c])
    for c in range(K):
        plt.plot(new_centroids[c][0],new_centroids[c][1], "s", markersize = 12, markerfacecolor = cluster_colors[c],label= 'Cluster %i' %(c+1),markeredgecolor = "black")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Spectral Clustering Result")
    plt.legend()


# In[5]:


B = np.array(euc_dist)
print(B.shape)
L = laplacian(B)
print(L.shape)


# In[6]:


eigval, eigvec = np.linalg.eig(L)
# eigvec = np.sort(eigvec)
Z = np.array(eigvec[:,0:5])
Z.shape


# - **Step 04.** Plot clusters  

# In[7]:


final_res= spectral_clustering(B, 5)
final_centroids= [np.mean(dataset[final_res== k], axis=0) for k in range(K)]


# In[8]:


plot_current_state(final_centroids, final_res, dataset)

