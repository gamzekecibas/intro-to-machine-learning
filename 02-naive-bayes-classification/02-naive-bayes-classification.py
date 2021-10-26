#!/usr/bin/env python
# coding: utf-8

# ### DASC 521: Intro to ML
# ### Homework 2: Naive Bayes Classification
# ### Gamze Keçibaş
# ---
# ### CONTENT
# - **Step 01.** Import libraries  
# - **Step 02.** Import images and labels  
#     - **Step 02.1** Train data visualization  
# - **Step 03.** Parameter Estimation  
#     - **Step 03.1** Calculate sample means  
#     - **Step 03.2** Calculate sample standard deviations  
#     - **Step 03.3** Calculate class priors    
# - **Step 04.** Model training  
#     - **Step 04.1** Confusion Matrix of trained model  
# - **Step 05.** Model testing  
# ---

# - **Step 01.** Import libraries

# In[1]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def safelog(x):
    return(np.log(x + 1e-100))


# - **Step 02.** Import images and labels

# In[2]:


images= np.genfromtxt('hw02_images.csv',delimiter=',').reshape(35000,784)
labels= np.genfromtxt('hw02_labels.csv',delimiter=',').reshape(35000)


# In[3]:


# First 30k images are using as training set
# Last 5k images are using as test set

train_images= images[:30000]
train_labels= labels[:30000] 
test_images= images[30000:]      
test_labels= labels[30000:]

print('Total set of images', images.shape)
print('Total set of labels', labels.shape)
print('Training set shape: ', train_images.shape)
print('Training label set shape: ', train_labels.shape)
print('Test set shape: ', test_images.shape)
print('Test label set shape: ', test_labels.shape)


# - **Step 02.** Import images and labels   
#     - **Step 02.1** Train data visualization  

# In[4]:


fig, axs = plt.subplots(2,10,figsize=(15,15), sharex= True, sharey=True)
n=0
for i in range(2):
    for j in range(10):
        axs[i,j].imshow(train_images[n].reshape(28,28),interpolation='none')
        n+=1
print('First 20 clothes in Training Set')
plt.show()


# - **Step 03.** Parameter Estimation  
#     - **Step 03.1** Calculate sample mean

# In[5]:


K= int(np.max(train_labels))
n= train_images.shape[0]

sample_means= [np.mean(train_images[train_labels == (c+1)], axis=0) for c in range(K)]
print('Sample means\n')
print('Size of sample means',len(sample_means),'x', len(sample_means[0]), '\n')
for i in range(len(sample_means)):
    print(f'Class [{i+1}]:')
    print(sample_means[i][:10])
    print('...')
    print(sample_means[i][-10:],'\n')


# - **Step 03.** Parameter Estimation  
#     - **Step 03.2** Calculate sample standard deviations

# In[6]:


sample_stdevs= [np.std(train_images[train_labels == (c+1)], axis=0) for c in range(K)]
print('Sample Standard Deviations\n')
print('Size of sample standard deviations',len(sample_stdevs),'x', len(sample_stdevs[0]), '\n')
for i in range(len(sample_stdevs)):
    print(f'Class [{i+1}]:')
    print(sample_stdevs[i][:10])
    print('...')
    print(sample_stdevs[i][-10:],'\n')


# - **Step 03.** Parameter Estimation  
#     - **Step 03.3** Calculate class priors

# In[7]:


class_priors= [np.mean(train_labels== (c+1)) for c in range(K)]
sum(class_priors)
print('Sample priors\n')
for i in range(len(class_priors)):
    print(f'Prior probability of Class [{i+1}]: ', class_priors[i])
print('\nTotal probability= ', sum(class_priors))


# - **Step 04** Model training

# In[8]:


y_pred_train = []

for i in range(train_images.shape[0]):
    scores= [(sum((-0.5*safelog(2*math.pi)) 
              - safelog(sample_means[c]) 
              - (train_images[i]-sample_means[c])*(train_images[i]-sample_means[c])/(2*sample_stdevs[c]*sample_stdevs[c]))) 
              + safelog(class_priors[c]) for c in range(K)]
    y_pred_train.append(np.argmax(scores)+1)
y_pred_train = np.array(y_pred_train)


# - **Step 04.** Model training
#     - **Step 04.1** Confusion Matrix of trained model

# In[9]:


CM_train = pd.crosstab(y_pred_train, train_labels, rownames = ['y_pred'], colnames = ['y_truth'])
print('Confusion Matrix of the Train Data:\n')
print(CM_train)


# - **Step 05.** Model testing

# In[10]:


y_pred_test = []

for i in range(test_images.shape[0]):
    scores= [(sum((-0.5*safelog(2*math.pi)) 
              - safelog(sample_means[c]) 
              - (test_images[i]-sample_means[c])*(test_images[i]-sample_means[c])/(2*sample_stdevs[c]*sample_stdevs[c]))) 
              + safelog(class_priors[c]) for c in range(K)]
    y_pred_test.append(np.argmax(scores)+1)
y_pred_test = np.array(y_pred_test)

CM_test = pd.crosstab(y_pred_test, test_labels, rownames = ['y_pred'], colnames = ['y_truth'])
print('Confusion Matrix of the Test Data:\n')
print(CM_test)

