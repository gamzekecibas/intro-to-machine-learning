#!/usr/bin/env python
# coding: utf-8

# ### DASC521: Intro to ML
# ### Homework 5: Decision Tree Regresion
# ### Gamze Keçibaş 60211  
# ---
# ### CONTENT
# - **Step 01.** Import libraries    
# - **Step 02.** Import dataset & Data preparation      

# * **Step 01.** Import libraries

# In[1]:


import math
import numpy as np
import matplotlib.pyplot as plt


# - **Step 02.** Import dataset & Data preparation

# In[2]:


DS= np.genfromtxt('hw05_data_set.csv',delimiter=',')
DS= np.delete(DS, (0), axis=0)

train_data= DS[:150,0]
train_label= DS[:150,1]
test_data= DS[150:,0]
test_label= DS[150:,1]

print('Size of given dataset: ', DS.shape)
print('First five rows: ')
print(DS[0:5])
print('Size of training data: ', train_data.shape)
print('First five rows: ')
print(train_data[0:5])
print('Size of training labels: ', train_label.shape)
print('First five rows: ')
print(train_label[0:5])
print('Size of test data: ', test_data.shape)
print('Size of test label: ', test_label.shape)

plt.show()
plt.figure(figsize = (10, 6))
plt.plot(train_data, train_label, "b.", markersize = 10, label = "training")
plt.plot(test_data, test_label, "r.", markersize = 10, label = "test")
plt.xlabel("Eruption Time [min]")
plt.ylabel("Waiting Time to Next Eruption [min]")
plt.title("Provided Dataset")
plt.legend(loc = "upper left")


# In[3]:


# get number of classes, number of samples, and number of features
K = np.max(DS[:,1])
N = DS.shape[0]
D= N

N_train = len(train_label)
N_test = len(test_label)


# In[4]:


def regression_tree(P):
    # create necessary data structures
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_splits = {}   
    node_means = {}   

    # put all training instances into the root node
    node_indices[1] = np.array(range(len(train_label)))
    is_terminal[1] = False
    need_split[1] = True
    
    # learning algorithm
    while True:
        split_nodes= [key for key, value in need_split.items() if value == True]
        if len(split_nodes) == 0:
            break
        # find the best split positions
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            node_mean = np.mean((train_label[data_indices]))      
    
            if len(train_data[data_indices]) <= P:
                is_terminal[split_node] = True
                node_means[split_node] =  node_mean
            else:
                is_terminal[split_node]= False                                                                
                unique_val= np.sort(np.unique(train_data[data_indices]))                                       
                split_pos = 0.5* (unique_val[1:len(unique_val)] + unique_val[0:(len(unique_val)-1)])       
                split_score = np.repeat(0, len(split_pos))
                
                for s in range(len(split_pos)):
                    err=0
                    right_inds = data_indices[train_data[data_indices] > split_pos[s]]    
                    left_inds = data_indices[train_data[data_indices] <= split_pos[s]]    
                    if len(left_inds) > 0:
                        err += np.sum((train_label[left_inds] - np.mean(train_label[left_inds]))**2)
                    if len(right_inds) > 0:
                        err += np.sum((train_label[right_inds] - np.mean(train_label[right_inds]))**2)
                    split_score[s]= err / (len(left_inds)+len(right_inds)+1)

                if len(unique_val) == 1:
                    is_terminal[split_node] = True
                    node_means[split_node] = node_mean
                    continue
                    
                best_split= split_pos[np.argmin(split_score)]
                node_splits[split_node]= best_split
                
                # creating left child
                left_inds = data_indices[train_data[data_indices] < best_split]
                node_indices[2*split_node]= left_inds
                is_terminal[2*split_node]= False
                need_split[2*split_node]= True
                
                # creating right child
                right_inds = data_indices[train_data[data_indices] >= best_split]
                node_indices[2* split_node +1]= right_inds
                is_terminal[2*split_node +1]= False
                need_split[2*split_node +1]= True
                
    return node_means, node_splits, is_terminal


# In[5]:


node_means, node_splits, is_terminal = regression_tree(25)

data_interval = np.linspace(np.minimum(np.min(train_data), np.min(test_data)), np.maximum(np.max(train_data), np.max(test_data)), 1000)


# In[6]:


# Traverse tree for test data points

def prediction(y, node_splits, node_means,  is_terminal):
    index = 1
    while(True):
        if is_terminal[index] == True:
            return node_means[index]
        else:
            if y <= node_splits[index]:
                index = 2*index
            else:
                index = 2*index + 1


# In[7]:


y_hat_test = [prediction(x,node_splits,node_means,is_terminal) for x in data_interval]
y_pred_test = np.array(y_hat_test)

y_hat_train = [prediction(x,node_splits,node_means,is_terminal) for x in data_interval]
y_pred_train = np.array(y_hat_test)


# In[8]:


left_borders = np.arange(0, 6, 0.01)
right_borders = np.arange(0 + 0.01, 6 + 0.01, 0.01)


plt.figure(figsize = (10, 6))
plt.plot(train_data, train_label, "b.", markersize = 10, label = "training")
plt.plot(test_data, test_label, "r.", markersize = 10, label = "test")
plt.plot(data_interval, y_pred_train,'k-',label= 'Prediction')

plt.xlabel("Eruption Time [min]")
plt.ylabel("Waiting Time to Next Eruption [min]")
plt.title("Decision Tree Prediction where P=25")
plt.title("P = 25")
plt.legend(loc = "upper left")
plt.show()


# In[9]:


def rmse(label, prediction):
    return np.sqrt(np.mean((label-prediction)**2))


# In[10]:


y_pred_train = [ prediction(x, node_splits, node_means, is_terminal) for x in train_data]
print("RMSE for training set= ",rmse(train_label, y_pred_train))

y_pred_test = [ prediction(x, node_splits, node_means, is_terminal) for x in test_data]
print("RMSE for test set= ",rmse(test_label, y_pred_test))


# In[13]:


P_array = np.arange(5, 55, 5)

rmse_train = []
rmse_test = []

for p in P_array:
    node_means, node_splits, is_terminal = regression_tree(p)
    y_pred_train = [ prediction(x, node_splits, node_means, is_terminal) for x in train_data]
    y_pred_test = [ prediction(x, node_splits, node_means, is_terminal) for x in test_data]
    
    rmse_train.append(rmse(train_label, y_pred_train))
    rmse_test.append(rmse(test_label, y_pred_test))


# In[14]:


plt.show()
plt.figure(figsize = (10, 6))
plt.plot(P_array, rmse_train, color='blue' ,marker="o")
plt.plot(P_array, rmse_test,color='red' ,marker="o")
plt.xlabel("P")
plt.ylabel("RMS Error")
plt.legend(["training", "test"])
plt.title('Different Pre-Pruning (P) Results')


# In[ ]:




