# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:59:05 2019

@author: DEVENDRA
"""
#%%
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
data=np.genfromtxt("Dataset_1_Team_3.csv", delimiter=",")
# visulalization of the data
colors = ['red','blue','green']
label=data[0:650,2]
#plt.scatter(data[:,0],data[:,1],c=label,cmap=matplotlib.colors.ListedColormap(colors))
w=np.zeros((3,3))
w_updated=w
table=np.zeros((3,3))
phi=np.zeros((1000,3))
phi[:,0:2]=data[:,0:2]
phi[:,2]=np.ones(1000)
phi=phi/100
# making up the target matrix
target=np.zeros((1000,3))
i=0
for row in data:
    if row[2]==0:
        target[i,0]=1
    elif row[2]==1:
        target[i,1]=1
    else:
        target[i,2]=1
    i=i+1
#%%
# y is posterior prob it is also N*K matrix
#def prob(weight,ph):
#    a0=weight[0,:].dot(ph)
#    a1=weight[1,:].dot(ph)
#    a2=weight[2,:].dot(ph)
#    s=math.exp(a0) + math.exp(a1) + math.exp(a2)
#    #if clas==0:
#    #    return  (math.exp(a0))/s
#    #elif clas==1:
#    #   return (math.exp(a1))/s
#    #else:
#    #    return (math.exp(a2))/s
#    an=[ (math.exp(a0))/s , (math.exp(a0))/s , (math.exp(a0))/s]
#    return np.array(an)
#%%
def prob1(weight,ph,clas):
    a=np.matmul(weight,ph)
    #print(a)
    s=math.exp(a[0]) + math.exp(a[1]) + math.exp(a[2])
    #print((math.exp(a[0]))/s,(math.exp(a[1]))/s,(math.exp(a[2]))/s)
    if clas==0:
        return  (math.exp(a[0]))/s
    elif clas==1:
        return (math.exp(a[1]))/s
    else:
        return (math.exp(a[2]))/s
#%%    
# defining loss function
#def loss_grad(t,w_upd,phii):
#    loss=0
#    for i in range(0,n):
#        loss=loss+(prob(w_upd,phii[i,:])-t[i,:]).dot(phii[i,:])
#        print(loss)
#    return loss

def loss_grad1(t,w_upd,phii,clas):
    loss=0
    for i in range(0,n):
        loss=loss+(prob1(w_upd,phii[i,:],clas)-t[i,clas])*(phii[i,:])
    return loss
#%%        
# training
n=650
#above  value of n gives the number of training points
eta=0.01
epochs=25
for i in range(0,epochs):
    w_updated[0,:]=w_updated[0,:]-eta*(loss_grad1(target,w_updated,phi,0))
    w_updated[1,:]=w_updated[1,:]-eta*(loss_grad1(target,w_updated,phi,1))
    w_updated[2,:]=w_updated[2,:]-eta*(loss_grad1(target,w_updated,phi,2))

# calculating the accuracy this is the total accuracy similarly do for train and test separately
accuracy=0
for i in range(0,650):
    prediction=np.matmul(w_updated,np.transpose(phi[i,:]))
    if prediction[0]<prediction[2] and prediction[1]<prediction[2]:
        if data[i,2]==2:
            accuracy+=1
    elif prediction[0]<prediction[1] and prediction[2]<prediction[1]:
        if data[i,2]==1:
            accuracy+=1
    elif prediction[1]<prediction[0] and prediction[2]<prediction[0]:
        if data[i,2]==0:
            accuracy+=1
accuracy_train=accuracy/650  
accuracy=0            
for i in range(650,1000):
    prediction=np.matmul(w_updated,np.transpose(phi[i,:]))
    if prediction[0]<prediction[2] and prediction[1]<prediction[2]:
        if data[i,2]==2:
            accuracy+=1
            table[2][2]+=1
        elif data[i,2]==1:
            table[2][1]+=1
        else:
            table[2][0]+=1
    elif prediction[0]<prediction[1] and prediction[2]<prediction[1]:
        if data[i,2]==1:
            accuracy+=1
            table[1][1]+=1
        elif data[i,2]==2:
            table[1][2]+=1
    elif prediction[1]<prediction[0] and prediction[2]<prediction[0]:
        if data[i,2]==0:
            accuracy+=1
            table[0][0]+=1
        elif data[i,2]==1:
            table[0][1]+=1
        else:
            table[0][2]+=1
    
accuracy_test=accuracy/350

#%%
# plotting the decision boundaries
t=1000
N=0.05
x = np.linspace(-3,9,t)
y = np.linspace(-2,1,t)
X, Y = np.meshgrid(x,y)
G_0_1=np.zeros((t,t))
G_1_2=np.zeros((t,t))
G_0_2=np.zeros((t,t))
G_0_1=np.matrix(G_0_1)
G_0_2=np.matrix(G_0_2)
G_1_2=np.matrix(G_1_2)
for i in range(t):
    for j in range(t):
        inp=[X[i,j],Y[i,j],0.01]
        inp=np.transpose(inp)
        prediction=np.matmul(w_updated,np.transpose(inp))
        if prediction[0]<prediction[1] and prediction[0]<prediction[2] and abs(prediction[1]-prediction[2])<N:
            G_1_2[i,j]=1
        if prediction[2]<prediction[1] and prediction[2]<prediction[0] and abs(prediction[1]-prediction[0])<N:
            G_0_1[i,j]=1
        if prediction[1]<prediction[0] and prediction[1]<prediction[2] and abs(prediction[0]-prediction[2])<N:
            G_0_2[i,j]=1
fig=plt.figure()
cp = plt.contour(X, Y, G_1_2,colors='black')
cp = plt.contour(X, Y, G_0_2,colors='black')
cp = plt.contour(X, Y, G_0_1,colors='black')
cp=plt.scatter(phi[0:650,0],phi[0:650,1],c=label,cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel('X1')
plt.ylabel('X2')
plt.plot([0],[0],'r',label="Class 0")
plt.plot([0],[0],'b',label="Class 1")
plt.plot([0],[0],'g',label="Class 2")
plt.legend(loc='best')
plt.title('Data set_1_num iter=25_etalearning=0.01')


    
        

        
        
