# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:51:51 2019

@author: DEVENDRA
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import preprocessing

data=np.genfromtxt("Dataset_4_Team_3.csv", delimiter=",")
label=data[:,2]
colors = ['red','green']
X_scaled = preprocessing.scale(data[:,0:2])
svclassifier=SVC(kernel='linear')
svclassifier.fit(X_scaled[0:650,:],label[0:650])
print(svclassifier.coef_)
y_pred_train=svclassifier.predict(X_scaled[0:650,:])
y_pred_test=svclassifier.predict(X_scaled[650:,:])
y_pred=svclassifier.predict(X_scaled)
#%%
# calculating the test error
accuracy=0
for i in range(650,1000):
    if y_pred[i]==label[i]:
        accuracy+=1
#%%
# understand this part really nice plot
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=label,cmap=matplotlib.colors.ListedColormap(colors))
plt.title('linear kernel')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 300)
yy = np.linspace(ylim[0], ylim[1], 300)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svclassifier.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])
ax.scatter(svclassifier.support_vectors_[:, 0], svclassifier.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.show()
#%%
# kernel support vector machines
# polynomial kernel
# why polynomial kernel not that good??
svclassifier1=SVC(kernel='poly',degree=3)
svclassifier1.fit(X_scaled[0:650,:],label[0:650])
y_pred_train=svclassifier1.predict(X_scaled[0:650,:])
y_pred_test=svclassifier1.predict(X_scaled[650:,:])
y_pred=svclassifier1.predict(X_scaled)
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=label,cmap=matplotlib.colors.ListedColormap(colors))
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 300)
yy = np.linspace(ylim[0], ylim[1], 300)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svclassifier1.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])
ax.scatter(svclassifier1.support_vectors_[:, 0], svclassifier1.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.show()
accuracy1=0
for i in range(650,1000):
    if y_pred[i]==label[i]:
        accuracy1+=1
#%%
# radial basic function kernel
svclassifier=SVC(kernel='rbf', random_state=0, gamma=10, C=10)
svclassifier.fit(X_scaled[0:650,:],label[0:650])
y_pred_train=svclassifier.predict(X_scaled[0:650,:])
y_pred_test=svclassifier.predict(X_scaled[650:,:])
y_pred=svclassifier.predict(X_scaled)
accuracy3=0
for i in range(650,1000):
    if y_pred[i]==label[i]:
        accuracy3+=1
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=label,cmap=matplotlib.colors.ListedColormap(colors))
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 300)
yy = np.linspace(ylim[0], ylim[1], 300)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svclassifier.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])
ax.scatter(svclassifier.support_vectors_[:, 0], svclassifier.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.show()
#%%
# need to study svm to understand and interpret these plots