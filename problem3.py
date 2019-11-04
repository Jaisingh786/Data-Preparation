# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:56:55 2019

@author: user
"""

from sklearn import svm 
import numpy as np
import math
import matplotlib

import matplotlib.pyplot as plt
from sklearn import preprocessing
 
data=np.genfromtxt("Dataset_5_Team_3.csv", delimiter=",")
label=data[0:650,2]
colors = ['red','green']    
data[:,0:2]= preprocessing.scale(data[:,0:2])#data[0:650,0:2]
svclassifier = svm.LinearSVC(C=1,max_iter=100)
svclassifier.fit(data[0:650,0:2],data[0:650,2])
t=np.hstack((svclassifier.coef_,svclassifier.intercept_[:,None]))
t=np.array(t[0])
margin=1/((t[0]*t[0]+t[1]*t[1])**0.5)
#margin=0.88
theta=(math.atan(-t[0]/t[1]))
plt.scatter(data[0:650,0],data[0:650,1],c=label,cmap=matplotlib.colors.ListedColormap(colors),s=6)
plt.title('Data set_linear_C=1_max_iter=100')
plt.xlabel('X1')
plt.ylabel('X2')
plt.plot([0],[0],'r',label="Class 0")
plt.plot([0],[0],'b',label="Class 1")
plt.legend(loc='best')
if(t[0]==0):
    y_intercept=-t[2]/t[1]
    axes=plt.axis()
    plt.plot([0,y_intercept],[y_intercept,y_intercept],'b--',linewidth=2)
    plt.xlim( [axes[0], axes[1]])
    plt.ylim( [axes[2], axes[3]])
else:
    x_intercept=-t[2]/t[0]
    y_intercept=-t[2]/t[1]
    axes=plt.axis()
    slope=-t[0]/t[1]
    plt.plot([axes[0],axes[1]],[slope*axes[0]+y_intercept,slope*axes[1]+y_intercept],'b',linewidth=2)
    plt.plot([axes[0],axes[1]],[slope*axes[0]+y_intercept+margin/math.cos(theta),slope*axes[1]+y_intercept+margin/math.cos(theta)],'m',linewidth=2)
    plt.plot([axes[0],axes[1]],[slope*axes[0]+y_intercept-margin/math.cos(theta),slope*axes[1]+y_intercept-margin/math.cos(theta)],'y',linewidth=2)
y_pred=[]
table=np.zeros((2,2))
for row in data:
    y_pred.append(svclassifier.predict([row[0:2]]))
y_pred=np.array(y_pred)
accuracy_test=0
accuracy_train=0
for i in range(650,1000):
    if(y_pred[i]==data[i,2]):
        accuracy_test+=1
        if(data[i,2]==0):
            table[0][0]+=1
        elif(data[i,2]==1):
            table[1][1]+=1
    else:
        if(data[i,2]==0):
            table[1][0]+=1
        elif(data[i,2]==1):
            table[0][1]+=1
for i in range(0,650):
    if(y_pred[i]==data[i,2]):
        accuracy_train+=1
accuracy_train/=650
accuracy_test/=350
#%%
#plt.scatter(data[0:650,0],data[0:650,1],c=label,cmap=matplotlib.colors.ListedColormap(colors))
#ax = plt.gca()
#xlim = ax.get_xlim()
#ylim = ax.get_ylim()
#xx = np.linspace(xlim[0], xlim[1], 300)
#yy = np.linspace(ylim[0], ylim[1], 300)
#YY, XX = np.meshgrid(yy, xx)
#xy = np.vstack([XX.ravel(), YY.ravel()]).T
#Z = svclassifier.decision_function(xy).reshape(XX.shape)
#ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])
###ax.scatter(svclassifier.support_vectors_[:, 0], svclassifier.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
#plt.show()
for i in range(0,650):
    if(abs(data[i,1]-slope*data[i,0]-y_intercept-margin/math.cos(theta))<0.05):
        #print(data[i,1],slope*data[i,0]+y_intercept+margin/math.cos(theta))
        plt.plot(data[i,0],data[i,1],'bo', markersize=4)
        #print(1)
    elif(abs(data[i,1]-slope*data[i,0]-y_intercept+margin/math.cos(theta))<0.05):
        plt.plot(data[i,0],data[i,1],'bo', markersize=4)
        #print(2)
plt.show()























