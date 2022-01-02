#!/usr/bin/env python
# coding: utf-8

# ### DASC521: Intro to ML
# ### Homework 7: Expectation - Maximization Clustering
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
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as Ellipse
import scipy.spatial as spa
from scipy.stats import multivariate_normal
import matplotlib.transforms as transforms


np.random.seed(421)


# - **Step 02.** Import dataset & Data preparation    

# In[ ]:


dataset = np.genfromtxt('hw07_data_set.csv',delimiter=',')
centroid_init = np.genfromtxt('hw07_initial_centroids.csv',delimiter=',')
print(dataset.shape)
print(centroid_init.shape)
X = dataset[:,0]
Y = dataset[:,1]

K= 5                                # number of cluster
N = 50                              # number of samples in classes except C5

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
plt.scatter(X,Y,color='black', label= 'Data Points')
for i, color in enumerate((["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"]), start=0):
    plt.plot(centroid_init[i,0], centroid_init[i,1], 'o', color= color, label= '$Initial Centroid #{n}$'.format(n=i+1))
plt.legend(loc='best')
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.title('Provided Dataset')
plt.legend()
plt.show()


# - **Step 03.** Define learning functions & clustering  

# In[ ]:


def update_centroids(memberships, X):
    if memberships is None:
        # initialize centroids
        new_centroids = X[np.random.choice(range(N), K, False),]
    else:
        # update centroids
        new_centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(new_centroids)

def update_memberships(new_centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(new_centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)

def plot_current_state(new_centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"])
    if memberships is None:
        plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])
    for c in range(K):
        plt.plot(new_centroids[c, 0], new_centroids[c, 1], "s", markersize = 12, 
                 markerfacecolor = cluster_colors[c], markeredgecolor = "black")
    plt.xlabel("x1")
    plt.ylabel("x2")
def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]


# In[ ]:


memberships = None
iteration = 1
while iteration <= 100:
    print("Iteration#{}:".format(iteration))

    old_centroids = centroid_init
    new_centroids = update_centroids(memberships, dataset)
    if np.alltrue(new_centroids == old_centroids):
        continue
    else:
        plt.figure(figsize = (12, 6))    
        plt.subplot(1, 2, 1)
        plot_current_state(new_centroids, memberships, dataset)

    old_memberships = memberships
    memberships = update_memberships(new_centroids, dataset)
    if np.alltrue(memberships == old_memberships):
        plt.show()
    else:
        plt.subplot(1, 2, 2)
        plot_current_state(new_centroids, memberships, dataset)
        plt.show()

    iteration = iteration + 1
print('Initial centroids: \n', centroid_init)
print('\nFinal centroids: \n', new_centroids)
clusters= [len(memberships[memberships==0]),len(memberships[memberships==1]),
           len(memberships[memberships==2]),len(memberships[memberships==3]),len(memberships[memberships==4])]
print(clusters)


# - **Step 04.** Plot clusters  

# In[ ]:


means= np.array([mu1,mu2,mu3,mu4,mu5])
covs= np.array([sigma1,sigma2,sigma3,sigma4,sigma5])

fig, ax = plt.subplots(figsize=(10,10),subplot_kw={'aspect': 'equal'})
plot_current_state(new_centroids, memberships, dataset)

# Original Gaussian distribution ellipses
for i in range(len(means)):

    mean = means[i]
    cov = covs[i] 
    if i<len(means):
        original_gaussian = np.random.multivariate_normal(mean, cov, N)
    else:
        original_gaussian = np.random.multivariate_normal(mean, cov, 2*N)
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    if i == len(means)-1:
        scale_x = 0.5*np.sqrt(cov[0, 0]) 
        mean_x = np.mean(X)
    else:
        scale_x = np.sqrt(cov[0, 0]) 
        mean_x = np.mean(X)

    # calculating the stdandard deviation of y ...
    if i == len(means)-1:
        scale_y = 0.5*np.sqrt(cov[1, 1])
        mean_y = np.mean(Y)
        width = 6
        height = 6
        ellipse = Ellipse(new_centroids[i,:], width, height, edgecolor= 'black',linestyle='--', facecolor= None, fill= False)
    else:
        scale_y = np.sqrt(cov[1, 1])
        mean_y = np.mean(Y)
        width = 2*np.sqrt(5.991*np.linalg.eig(cov)[0][0])
        height = 2*np.sqrt(5.991*np.linalg.eig(cov)[0][1])
        ellipse = Ellipse(new_centroids[i,:], width, height, edgecolor= 'black',linestyle='--', facecolor= None, fill= False)
    
    angles= [30,30,-30,-30,0]
    transf = transforms.Affine2D()     .rotate_deg_around(new_centroids[i,0],new_centroids[i,1],angles[i])     .scale(scale_x, scale_y)     .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    ax.add_artist(ellipse)

# Final distribution ellipses
for i in range(len(means)):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928"])
    
    mean = means[i]
    cov = covs[i] 
    color = cluster_colors[i]
    if i<len(means):
        original_gaussian = np.random.multivariate_normal(mean, cov, N)
    else:
        original_gaussian = np.random.multivariate_normal(mean, cov, 2*N)
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    if i == len(means)-1:
        scale_x = 0.5*np.sqrt(cov[0, 0]) 
        mean_x = np.mean(X)
    else:
        scale_x = np.sqrt(cov[0, 0]) 
        mean_x = np.mean(X)

    # calculating the stdandard deviation of y ...
    if i == len(means)-1:
        scale_y = 0.5*np.sqrt(cov[1, 1])
        mean_y = np.mean(Y)
        width = 4
        height = 4
        ellipse_1 = Ellipse(new_centroids[i,:], width, height, edgecolor= color,linestyle='-', facecolor= None, fill= False)
    else:
        scale_y = np.sqrt(cov[1, 1])
        mean_y = np.mean(Y)
        width = 2*np.sqrt(5.991*np.linalg.eig(cov)[0][0])
        height = 2*np.sqrt(5.991*np.linalg.eig(cov)[0][1])
        ellipse_1 = Ellipse(new_centroids[i,:], width, height, edgecolor= color,linestyle='-', facecolor= None, fill= False)
    
    angles= [30,30,-30,-30,0]
    transf = transforms.Affine2D()     .rotate_deg_around(new_centroids[i,0],new_centroids[i,1],angles[i])
    ellipse_1.set_transform(transf + ax.transData)
    ax.add_artist(ellipse_1)
plt.title('Final Clustering')

plt.show()

