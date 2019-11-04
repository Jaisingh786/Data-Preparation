# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 01:53:22 2019

@author: DEVENDRA
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 14:55:26 2019

@author: DEVENDRA
"""

# kernel logistic regression
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
#%%
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
# importing dataset
data=np.genfromtxt("Dataset_2_Team_3.csv", delimiter=",")
phi=data[:,0:2]
#phi=(data[:,0:2])/1000
colors = ['red','green']
for row in data:
    if(row[2]==0):
        row[2]=-1
    else:
        row[2]=1
label=data[0:650,2]
plt.scatter(data[0:650,0],data[0:650,1],c=label,cmap=matplotlib.colors.ListedColormap(colors))
#%%
# calculating the linear kernel
k_linear=np.matmul(phi,np.transpose(phi))
# calculating the polynomial kernel
k_poly=np.zeros((1000,1000))
for i in range(1000):
    for j in range(1000):
        k_poly[i,j]=(np.matmul(phi[i,:],phi[j,:]))**3
# calculating the gaussian kernel
k_rbf=np.zeros((1000,1000))
sigma=1
for i in range(1000):
    for j in range(1000):
        k_rbf[i,j]=np.exp((-(np.linalg.norm(phi[i,:]-phi[j,:]))**2)/(2*(sigma**2)))

#%%
# n is no of data points

# calculating the loss_gradient 
def loss_gradient(K,alpha,Y):
    L=np.zeros(1000)
    L=np.array(L)
    #lamb=0
    a=0
    n=650
#    print(alpha.shape)
#    print(Y.shape)
#    print(K.shape)
    for i in range(n):
        a=np.exp(-(Y[i])*(np.matmul(np.transpose(alpha),K[:,i])))
        #print(a)
        #L=L+((a/(n*(1+a)))*(-Y[i]*K[:,i]))+(lamb*np.matmul(K,alpha))
        #print(((a/(n*(1+a))*(-Y[i]*K[:,i]))).shape)
        L=L+(sig(-a))*(-Y[i]*K[:,i])
        #print(L.shape)
    #return np.transpose(L)
    return L
#%%
# defining alpha which n*1 vector where n is the number of data points
alpha_linear=np.zeros(1000)
alpha_poly=np.zeros(1000)
alpha_rbf=np.zeros(1000)
eta1=1
#eta2=0.001
eta3=1
no_of_iter=200
#a=math.exp(-(data[1,2])*(np.matmul(np.transpose(alpha_linear),k_linear[:,5])))
#L=a/(10*(1+a))*(-data[1,2]*k_linear[:,5])
#a=np.array(a)

for i in range(no_of_iter):
    alpha_linear=alpha_linear-eta1*(loss_gradient(k_linear,alpha_linear,data[:,2]))/650
    alpha_poly=alpha_poly-eta3*(loss_gradient(k_poly,alpha_poly,data[:,2]))/650
    #alpha_rbf=alpha_rbf-eta3*(loss_gradient(k_rbf,alpha_rbf,data[:,2]))/650
for i in range(10):
    alpha_rbf=alpha_rbf-0.0001*(loss_gradient(k_rbf,alpha_rbf,data[:,2]))/650
    #loss_gradient(k_linear,alpha_linear,data[:,2])
# training is dones above
#%%
# calculating accuracy
accuracy_linear=0
accuracy_poly=0
accuracy_rbf=0
for i in range(0,650):
    a=np.matmul(np.transpose(alpha_linear),k_linear[:,i])
    b=np.matmul(np.transpose(alpha_poly),k_poly[:,i])
    c=np.matmul(np.transpose(alpha_rbf),k_rbf[:,i])
    prediction=sig(a)
    #print(prediction,1)
    prediction1=sig(b)
    #print(prediction1,2)
    prediction2=sig(c)
    #print(c,prediction2,3)
    if prediction>0.5 and data[i,2]==1:
        accuracy_linear=accuracy_linear+1
    if prediction<0.5 and data[i,2]==-1:
        accuracy_linear=accuracy_linear+1
    if prediction1>0.5 and data[i,2]==1:
        accuracy_poly=accuracy_poly+1
    if prediction1<0.5 and data[i,2]==-1:
        accuracy_poly=accuracy_poly+1
    if prediction2>0.5 and data[i,2]==1:
        accuracy_rbf=accuracy_rbf+1
    if prediction2<0.5 and data[i,2]==-1:
        accuracy_rbf=accuracy_rbf+1
accuracy_linear_train=accuracy_linear/6.5
accuracy_poly_train=accuracy_poly/6.5
accuracy_rbf_train=accuracy_rbf/6.5
#%%
accuracy_linear=0
accuracy_poly=0
accuracy_rbf=0
for i in range(650,1000):
    a=np.matmul(np.transpose(alpha_linear),k_linear[:,i])
    b=np.matmul(np.transpose(alpha_poly),k_poly[:,i])
    c=np.matmul(np.transpose(alpha_rbf),k_rbf[:,i])
    prediction=sig(a)
    #print(prediction,1)
    prediction1=sig(b)
    #print(prediction1,2)
    prediction2=sig(c)
    #print(c,prediction2,3)
    if prediction>0.5 and data[i,2]==1:
        accuracy_linear=accuracy_linear+1
    if prediction<0.5 and data[i,2]==-1:
        accuracy_linear=accuracy_linear+1
    if prediction1>0.5 and data[i,2]==1:
        accuracy_poly=accuracy_poly+1
    if prediction1<0.5 and data[i,2]==-1:
        accuracy_poly=accuracy_poly+1
    if prediction2>0.5 and data[i,2]==1:
        accuracy_rbf=accuracy_rbf+1
    if prediction2<0.5 and data[i,2]==-1:
        accuracy_rbf=accuracy_rbf+1
accuracy_linear_test=accuracy_linear/3.5
accuracy_poly_test=accuracy_poly/3.5
accuracy_rbf_test=accuracy_rbf/3.5
  
#%%
# plotting the decision boundaries

#plt.scatter(data[0:650,0],data[0:650,1],c=label,cmap=matplotlib.colors.ListedColormap(colors))
#plt.plot([0],[0],'r',label='Class 0')
#plt.plot([0],[0],'g',label='Class 1')
#plt.title('Radial basis function kernel Parameter C=10 and Gamma=1')
#plt.xlabel('x1')
#plt.xlabel('x2')
#plt.legend(loc='best')
#t=1000
#N=0.05
#x = np.linspace(-2,2.8,t)
#y = np.linspace(-1.5,2,t)
#X, Y = np.meshgrid(x,y)
#G_0_1=np.zeros((t,t))
#G_1_2=np.zeros((t,t))
#G_0_2=np.zeros((t,t))
#G_0_1=np.matrix(G_0_1)
#G_0_2=np.matrix(G_0_2)
#G_1_2=np.matrix(G_1_2)
#
#for i in range(t):
#    for j in range(t):

#%%
t=100
xx=np.linspace(-2,3,t)
yy=np.linspace(-1.5,2,t)
X, Y = np.meshgrid(xx,yy)
N=0.1
G_lin=np.zeros((t,t))
#G_poly=np.zeros(1000)
#G_rbf=np.zeros(1000)
K_=np.zeros(1000)

for j in range(0,100):
    for k in range(0,100):
        for i in range(1000):
            K_[i]=np.exp(-(np.linalg.norm(phi[i,:]-np.array([X[j,k],Y[j,k]]))**2)/(2*(sigma**2)))
            #print(K_[i])
        a=np.matmul(np.transpose(alpha_linear),K_)
       # b=np.matmul(np.transpose(alpha_poly),k_poly[:,i])
       # c=np.matmul(np.transpose(alpha_rbf),k_rbf[:,i])
        print(a)
        prediction=sig(a)
        prediction1=1-prediction
        #print(prediction,prediction1)
        if abs(prediction-prediction1)<0.1:
            G_lin[j][k]=1
    #if abs(b)<N:
    #    G_poly[i]=1
   # if abs(c)<N:
   #     G_rbf[i]=1        
cp = plt.contour(X,Y,G_lin,colors='black')
plt.scatter(data[0:650,0],data[0:650,1],c=label,cmap=matplotlib.colors.ListedColormap(colors))
plt.plot([0],[0],'r',label='Class 0')
plt.plot([0],[0],'g',label='Class 1')
plt.title('Radial basis function kernel Parameter C=10 and Gamma=1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='best')    
    

        
    
    
        
