# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:14:20 2019

@author: user
"""
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
data=np.genfromtxt("Dataset_4_Team_3.csv", delimiter=",")
table=np.zeros((2,2))
table2=np.zeros((2,2))
label=data[:,2]
colors = ['red','green']
plt.scatter(data[:,0],data[:,1],c=label,cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel('X1')
plt.ylabel('X2')
plt.plot([0],[0],'r',label="Class 0")
plt.plot([0],[0],'g',label="Class 1")
plt.legend(loc='best')
plt.title('Data set_4_num iter=100_etalearning=1')
def sig(x):
    if(x<-700):
        return 0
    elif(x==0):
        return 0.5
    else:
        return 1/(1+math.exp(-x))
W_in=[0,0,0]
eta=1
epochs=100
W_updated=W_in
W_updated=np.array(W_updated)
for i in range(0,epochs):
    for row in data:
        row2=[0.0,0.0,0.0]
        row2[0:2]=row[0:2]
        row2[2]=1
        row2=np.array(row2)
        #print(row2,row)
        W_updated=W_updated-eta*(sig(W_updated.dot(row2))-row[2])*row2
        #print(W_updated)
accuracy=result=0
for row in data[0:650]:
    row2=[0.0,0.0,0.0]
    row2[0:2]=row[0:2]
    row2[2]=1
    row2=np.array(row2)
    prediction=sig(W_updated.dot(row2))
    if(prediction>0.5):
        result=1
    if(prediction<0.5):
        result=0
    if(result==row[2]):
        accuracy+=1
accuracy_train=(accuracy/650)*100
accuracy=result=0
for row in data[650:]:
    row2=[0.0,0.0,0.0]
    row2[0:2]=row[0:2]
    row2[2]=1
    row2=np.array(row2)
    prediction=sig(W_updated.dot(row2))
    if(prediction>0.5):
        result=1
    if(prediction<0.5):
        result=0
    if(result==row[2]):
        accuracy+=1
        if(result==0):
            table[0][0]+=1
        if(result==1):
            table[1][1]+=1
    else:
        if(result==0):
            table[0][1]+=1
        else:
            table[1][0]+=1
accuracy_test=(accuracy/350)*100
if(W_updated[0]==0):
    y_intercept=-W_updated[2]/W_updated[1]
    axes=plt.axis()
    plt.plot([0,y_intercept],[y_intercept,y_intercept],'b--',linewidth=2)
    plt.xlim( [axes[0], axes[1]])
    plt.ylim( [axes[2], axes[3]])
else:
    x_intercept=-W_updated[2]/W_updated[0]
    y_intercept=-W_updated[2]/W_updated[1]
    axes=plt.axis()
    slope=-y_intercept/x_intercept
    plt.plot([axes[0],axes[1]],[slope*axes[0]+y_intercept,slope*axes[1]+y_intercept],'k',linewidth=2)
#%%
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

def sig(x):
    #print(x)
    if(x<-700):
        return 0
    elif(x>700):
        return 1
    elif(x==0):
        return 0.5
    else:
        return 1/(1+math.exp(-x))

data=np.genfromtxt("Dataset_2_Team_3.csv", delimiter=",")
W_in=[0,0,0]
table2=np.zeros((2,2))

label=data[:,2]
colors = ['red','green']
plt.scatter(data[:,0],data[:,1],c=label,cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel('X1')
plt.ylabel('X2')
plt.plot([0],[0],'r',label="Class 0")
plt.plot([0],[0],'g',label="Class 1")
plt.legend(loc='best')
plt.title('Data set_2_lamda=100_num iter=1_etalearning=0.001')

lam=100
eta=0.001
epochs=1
W_updated=W_in
W_updated=np.array(W_updated)
for i in range(0,epochs):
    for row in data:
        row2=[0.0,0.0,0.0]
        row2[0:2]=row[0:2]
        row2[2]=1
        row2=np.array(row2)
        #print(W_updated.dot(row2))
        W_updated=W_updated-eta*((sig(W_updated.dot(row2))-row[2])*row2+lam*W_updated)
        #print(W_updated)
accuracy_reg_train=result=0
for row in data[0:650]:
    row2=[0.0,0.0,0.0]
    row2[0:2]=row[0:2]
    row2[2]=1
    row2=np.array(row2)
    prediction=sig(W_updated.dot(row2))
    if(prediction>0.5):
        result=1
    if(prediction<0.5):
        result=0
    if(result==row[2]):
        accuracy_reg_train+=1
accuracy_reg_train=(accuracy_reg_train/650)*100
accuracy_reg_test=result=0
for row in data[650:]:
    row2=[0.0,0.0,0.0]
    row2[0:2]=row[0:2]  
    row2[2]=1
    row2=np.array(row2)
    prediction=sig(W_updated.dot(row2))
    #print(prediction)
    if(prediction>0.5):
        result=1
    if(prediction<0.5):
        result=0
    if(result==row[2]):
        accuracy_reg_test+=1
        if(result==0):
            table2[0][0]+=1
        if(result==1):
            table2[1][1]+=1
    else:
        if(result==0):
            table2[0][1]+=1
        else:
            table2[1][0]+=1
accuracy_reg_test=(accuracy_reg_test/350)*100
if(W_updated[0]==0):
    y_intercept=-W_updated[2]/W_updated[1]
    axes=plt.axis()
    plt.plot([0,y_intercept],[y_intercept,y_intercept],'b--',linewidth=2)
    plt.xlim( [axes[0], axes[1]])
    plt.ylim( [axes[2], axes[3]])
else:
    x_intercept=-W_updated[2]/W_updated[0]
    y_intercept=-W_updated[2]/W_updated[1]
    axes=plt.axis()
    slope=-y_intercept/x_intercept
    plt.plot([axes[0],axes[1]],[slope*axes[0]+y_intercept,slope*axes[1]+y_intercept],'k',linewidth=2)
#%%






















  
    
    

