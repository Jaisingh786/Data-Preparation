# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:53:07 2019

@author: user
"""

import numpy as np

train=np.genfromtxt("train.csv", delimiter=",")
train=train[1:,:]
test=np.genfromtxt("test.csv", delimiter=",")
test=test[1:,:]
attr_relevance=np.genfromtxt("genome_scores.csv", delimiter=",")
attr_relevance=attr_relevance[1:,:]
W_trained_users=[]
rating =[]
movie=[]
tag=[]
#%%
c=test[0][0]
A_test=[]
A_test.append(c)
for row in test:
    if(row[0]==c):
        pass
    else:
        c=row[0]
        A_test.append(c)
A_train=[]
c=train[0][0]
A_train.append(c)
for row in train:
    if(row[0]==c):
        pass
    else:
        c=row[0]
        A_train.append(c)
a=b=c=0
needed_train=[]
only_in_test=[]
C=np.ones((10000,1))
for i in A_train:
    C[int(i)]-=3   
for i in A_test:
    C[int(i)]+=2
for i in range(0,10000):
    if(C[int(i)]==-2):
        a+=1#only in train
    elif(C[int(i)]==3):
        b+=1
        only_in_test.append(i)#only in test
    elif(C[int(i)]==0):
        needed_train.append(i)
        c+=1
needed_train=np.array(needed_train)
only_in_test=np.array(only_in_test)
#%%
#findout how many movies have tags
#import pandas as pd
#tags= pd.read_csv("tags.csv")
#tags=tags.sort_values('movieId') 
#for row in train:
#    if(row[0]==1):#a givrn user
#        movie.append(int(row[1]))
#        rating.append(row[2])
#        taged=[]
#        for j in range(0,tags.shape[0]):
#            temp=j
#            if(tags.iloc[temp,0]==movie[-1]):
#                while(tags.iloc[temp,1]==movie[-1]):
#                    taged.append(tags[temp,1])
#                    temp+=1
#                #print(i)
#                if(len(taged)!=0):
#                    #print(len(taged))
#                    break
#        if(len(taged)!=0):
#           tag.append(taged)
#           break
#    else:
#         break
#%%
weights=[]
y=[]
A=[]
store_train=[]
lam=0.001
for i in range(0,train.shape[0]):
    if(i in needed_train):
        temp=train[i,0]#tempindicates user name
        store_train.append(temp)
        while(temp==train[i,0]):#this loops store all the movies he rated into A
            A.append((attr_relevance[attr_relevance[:,0]==train[i,1]])[:,2])
            y.append(train[i,2])
            i+=1
        A=np.matrix(A)
        y=np.matrix(y)
        y=np.transpose(y)
    #A=np.transpose(A)
        i=i-1
        c=np.ones((A.shape[0],1))
        A=np.hstack((A,c))
        res=(np.linalg.inv((np.transpose(A))*A+2*lam*(np.identity(A.shape[1]))))*(np.transpose(A))*y
        weights.append(res)
        A=[]
        y=[]#train.shape[0]  
store_train=np.array(store_train)
#%%
for i in range(0,530):
    weights[i]=np.array(weights[i])
    #x=np.matrix(weights[i])
    #x=np.transpose(x)
    #print(x[0,0:1128].dot(attr_relevance[attr_relevance[:,0]==0][:,2])+x[0,1128])    
weights=np.matrix(weights)
np.savetxt("weights.csv",weights, delimiter=",")
np.savetxt("foo.csv", store_train, delimiter=",")
#Weigths=np.genfromtxt("weights.csv", delimiter=",")


























    
    
            
        
                
                
                
        
    