# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:19:39 2019

@author: user
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
data=np.genfromtxt("Dataset_4_Team_3.csv", delimiter=",")
mean1=np.mean(data[:,0])
mean2=np.mean(data[:,1])
var1=np.var(data[:,0])
var2=np.var(data[:,1])
for row in data:
    row[0]=(row[0]-mean1)/var1
    row[1]=(row[1]-mean2)/var2
label=data[0:650,2]
colors = ['red','green']
plt.scatter(data[0:650,0],data[0:650,1],c=label,cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel('X1')
plt.ylabel('X2')
plt.plot([0],[0],'r',label="Class 0")
plt.plot([0],[0],'b',label="Class 1")
plt.legend(loc='best')
plt.title('Data set_4_num iter=1_etalearning=1')
W_in=[0.0,0.0,0.0]
eta=1
W_updated=W_in
W_updated=np.array(W_updated)
epochs=1
table=np.zeros((2,2))
for i in range(0,epochs):
    for row in data[0:650]:
        row2=[0.0,0.0,0.0]
        row2[0:2]=row[0:2]
        row2[2]=1
        row2=np.array(row2)
        if(row[2]==1 and W_updated.dot(row2)<=0):
            W_updated+=eta*row2
        elif(row[2]==0 and W_updated.dot(row2)>=0):
            W_updated-=eta*row2
accuracy_train=result=accuracy_test=0
for row in data[0:650]:
    row2=[0.0,0.0,0.0]
    row2[0:2]=row[0:2]
    row2[2]=1
    row2=np.array(row2)
    prediction=W_updated.dot(row2)
    if(prediction>0):
        result=1
    if(prediction<0):
        result=0
    if(result==row[2]):
        accuracy_train+=1
accuracy_train=(accuracy_train/650)*100

for row in data[650:]:
    row2=[0.0,0.0,0.0]
    row2[0:2]=row[0:2]
    row2[2]=1
    row2=np.array(row2)
    prediction=W_updated.dot(row2)
    if(prediction>0):
        result=1
    if(prediction<0):
        result=0
    if(result==row[2]):
        accuracy_test+=1
        if(result==0):
            table[0][0]+=1
        if(result==1):
            table[1][1]+=1
    else:
        if(result==0):
            table[0][1]+=1
        else:
            table[1][0]+=1
accuracy_test=(accuracy_test/350)*100
t=W_updated
x_intercept=-t[2]/t[0]
y_intercept=-t[2]/t[1]
if(x_intercept!=0 and y_intercept!=0):
    axes=plt.axis()
    slope=-y_intercept/x_intercept
    plt.plot([axes[0],axes[1]],[slope*axes[0]+y_intercept,slope*axes[1]+y_intercept],'b--',linewidth=2)
if(t[2]==0):
    axes=plt.axis()
    slope=-t[0]/t[1]
    plt.plot([axes[0],axes[1]],[slope*axes[0],slope*axes[1]],'b--',linewidth=2)    
#%%
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
ray_data=np.load('ray.npy')
c=np.zeros((1000,1))
ray_data=np.hstack((ray_data,c))
chi_data=np.load('chicken.npy')
c=np.ones((1000,1))
chi_data=np.hstack((chi_data,c))
tot_data= np.concatenate((ray_data,chi_data), axis=0)
#tot_data=np.matrix(tot_data)
random.shuffle(tot_data)
tot_train=tot_data[0:1300]
tot_test=tot_data[1300:]

#label=tot_data[:,2048]
#colors = ['red','green']
#plt.scatter(tot_data,tot_data,c=label,cmap=matplotlib.colors.ListedColormap(colors))
#plt.xlabel('X1')
#plt.ylabel('X2')
#plt.plot([0],[0],'r',label="Class 0")
#plt.plot([0],[0],'b',label="Class 1")
#plt.legend(loc='best')
#plt.title('Image_dataset_num iter=1_etalearning=1')
W_in=np.zeros(2049)
#(W_in).T
eta=1
W_updated=W_in
W_updated=np.array(W_updated)
epochs=2
#table=np.zeros((2,2))
for i in range(0,epochs):
    for row in tot_train:
        row2=np.zeros(2049)
        #row2.T
        row2[0:2048]=row[0:2048]
        row2[2048]=1
        row2=np.array(row2)
        if(row[2048]==1 and W_updated.dot(row2)<=0):
            W_updated+=eta*row2
        elif(row[2048]==0 and W_updated.dot(row2)>=0):
            W_updated-=eta*row2

accuracy_train=result=accuracy_test=0
for row in tot_train:
    row2=np.zeros(2049)
    row2[0:2048]=row[0:2048]
    row2[2048]=1
    row2=np.array(row2)
    prediction=W_updated.dot(row2)
    #print(prediction)
    if(prediction>0):
        result=1
    if(prediction<0):
        result=0
    if(result==row[2048]):
        accuracy_train+=1
accuracy_train=(accuracy_train/1300)*100

key=0
for row in tot_test:
    row2=np.zeros(2049)
    row2[0:2048]=row[0:2048]
    row2[2048]=1
    row2=np.array(row2)
    prediction=W_updated.dot(row2)
    #print(prediction)
    if(prediction>0):
        result=1
    if(prediction<0):
        result=0
    if(result==row[2048]):
        accuracy_test+=1
        if(row[2048]==0):
            key+=1
accuracy_test=(accuracy_test/700)*100



















