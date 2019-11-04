# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:53:48 2019

@author: DEVENDRA
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
data=np.genfromtxt("Dataset_2_Team_3.csv", delimiter=",")
# visulalization of the data
colors = ['red','green','blue']
label=data[:,2]
plt.scatter(data[:,0],data[:,1],c=label,cmap=matplotlib.colors.ListedColormap(colors))
# have to feature map
#%%
# below n gives the number of datapoints
n=1000
# note last entery in below statement contain the class of that feature vector
phi=[np.ones(1000),data[:,0],data[:,1],(data[:,0])**2,(data[:,1])**2,(data[:,0])*(data[:,1]),data[:,2]]
ph=[]
for j in range(7):
    for i in range(n):
        ph.append((phi[j])[i])
ph=np.array(ph)
ph.resize((7,1000))
ph=np.transpose(ph)
# in above lines got a new feature vector
#%%
# writing functions which will be used in logistic regressions
def sig(x):
    if(x<-700):
        return 0
    elif(x>700):
        return 1
    elif(x==0):
        return 0.5
    else:
        return 1/(1+math.exp(-x))

#%%
# writing the update rule for weights using gradient descent
W_in=[1,1,1,1,1,1]
eta=0.05
epochs=50
W_updated=W_in
W_updated=np.array(W_updated)
for i in range(0,epochs):
    for row in ph:
        row1=row[0:6]
#        if row[6]==1:
#            row[6]=0
#        else:
#            row[6]=1
        W_updated=W_updated-eta*(sig(W_updated.dot(row1))-row[6])*row1
        
#%%
# calculating the accuracy
# doubt row[6]==1 means which class to compare now
train_accuracy=0
test_accuracy=0
for row in ph[0:650,:]:
    prediction1=sig(W_updated.dot(row[0:6]))
    if prediction1>0.5 and row[6]==1:
        train_accuracy=train_accuracy+1
    if prediction1<=0.5 and row[6]==0:
        train_accuracy=train_accuracy+1
for row in ph[650:1000,:]:
    prediction1=sig(W_updated.dot(row[0:6]))
    if prediction1>0.5 and row[6]==1:
        test_accuracy=test_accuracy+1
    if prediction1<=0.5 and row[6]==0:
        test_accuracy=test_accuracy+1
#%%
# 6 degree transformed space
# below n gives the number of datapoints
n=1000
phil=[]
# note last entery in below statement contain the class of that feature vector
phi1=[np.ones(1000),data[:,0],data[:,1],(data[:,0])**2,(data[:,1])**2,(data[:,0])*(data[:,1]),(data[:,0])**3,(data[:,1])**3,((data[:,0])**2)*(data[:,1]),((data[:,1])**2)*(data[:,0]),(data[:,0])**4,(data[:,1])**4,((data[:,0])**3)*(data[:,1]),((data[:,1])**3)*(data[:,0]),((data[:,0])**2)*((data[:,1])**2),(data[:,0])**5,(data[:,1])**5,((data[:,0])**4)*(data[:,1]),((data[:,1])**4)*(data[:,0]),((data[:,0])**2)*((data[:,1])**3),((data[:,0])**3)*((data[:,1])**2),(data[:,0])**6,(data[:,1])**6,((data[:,0])**5)*(data[:,1]),((data[:,1])**5)*(data[:,0]),((data[:,0])**3)*((data[:,1])**3),((data[:,0])**4)*((data[:,1])**2),((data[:,0])**2)*((data[:,1])**4),data[:,2]]
# doing the appropriate transpose to get the feature vector
ph1=[]
for j in range(29):
    for i in range(n):
        ph1.append((phi1[j])[i])
ph1=np.array(ph1)
ph1.resize((29,1000))
ph1=np.transpose(ph1)
# in above lines got a new feature vector
#%%
W_in1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
eta=0.05
epochs=50
W_updated1=W_in1
W_updated1=np.array(W_updated1)
for i in range(0,epochs):
    for row in ph1:
        row1=row[0:28]
#        if row[6]==1:
#            row[6]=0
#        else:
#            row[6]=1
        W_updated1=W_updated1-eta*(sig(W_updated1.dot(row1))-row[28])*row1
#%%
# Calculating the accuracy
train_accuracy1=0
test_accuracy1=0
for row in ph1[0:650,:]:
    prediction1=sig(W_updated1.dot(row[0:28]))
    if prediction1>0.5 and row[28]==1:
        train_accuracy1=train_accuracy1+1
    if prediction1<=0.5 and row[28]==0:
        train_accuracy1=train_accuracy1+1
for row in ph1[650:1000,:]:
    prediction1=sig(W_updated1.dot(row[0:28]))
    if prediction1>0.5 and row[28]==1:
        test_accuracy1=test_accuracy1+1
    if prediction1<=0.5 and row[28]==0:
        test_accuracy1=test_accuracy1+1
#%%
        