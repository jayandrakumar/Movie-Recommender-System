#!/usr/bin/python3

## knn template for making movie predictions utilizing
## an adapted version of the k-nearest neighbors algorithm
## This template provides examples of:
##    1) How to read parameters from the command line in python
##    2) how to type-cast (changing the value of k on the command
##       line from a string to an integer)
##    3) how to parse/split input lines
##    4) how to build the ratings matrix


## arg1 -- rating file (training set)
## arg2 -- ratings to predict (test set)
## arg3 -- predictions file (output)
## arg4 -- k 


import timeit
start =timeit.default_timer()
import sys
import numpy as np
import math
import csv
from decimal import Decimal, ROUND_UP

## Offsets in ratings file for data
USERIDX=0
MOVIEIDX=1
RATINGIDX=2

if len(sys.argv) != 5:
	print('usage: trainingDataFileName testDataFileName predictionsFile k')
	sys.exit(99)

trainingFileName = sys.argv[1]
testFileName     = sys.argv[2]
outFileName      = sys.argv[3]
k                = int(sys.argv[4])



#########################
## Main
#########################


## read in training data
trainingFile=open(trainingFileName)
trainingData=trainingFile.readlines()
trainingFile.close()

## read in test data
testFile = open(testFileName)
testData=testFile.readlines()
testFile.close()




## Get Max userid and movieID so we can 
## build a U by M ratings matrix

userCount=0
movieCount=0
for r in trainingData:
        rLine = r.split(',')
        userID=int(rLine[USERIDX])
        movieID=int(rLine[MOVIEIDX])
        if userID > userCount:
                userCount = userID
        if movieID > movieCount:
                movieCount = movieID

## now lets construct a UxM matrix of all zeros

userIDMax=userCount   + 1 # userCount is zero based
movieIDMax=movieCount + 1 

R = np.zeros(shape=(userIDMax,movieIDMax))

## build ratings matrix
for r in trainingData:
	rLine = r.split(',')

	userID = int(rLine[USERIDX])
	movieID= int(rLine[MOVIEIDX])
	rating = float(rLine[RATINGIDX])

	R[userID][movieID]  = float(rating)

## If you want to debug this process, uncomment
## the line below so you can see how it is processed
##	print('userid:' + str(userID) + ' movieID:' + 
##          str(movieID) + ' rating:' + str(R[userID][movieID]))

#creating a matrix to store euclidean distances.
euclidean_distance=np.full((userIDMax,userIDMax),-2,dtype='f')

#defining a function to calculate euclidean distance
def euc_distance(x,y):
    return math.sqrt(sum([(a-b)**2 for a,b in zip(x,y)]))

#calculating the euclidean distances
for i in range(userIDMax):
        print(i)
        for j in range(userIDMax):
                if euclidean_distance[i][j]==-2:
                        a=euc_distance(R[i],R[j])
                        euclidean_distance[i][j]=a
                        euclidean_distance[j][i]=a

#saving the data into numpy file
#np.save("euclidean_distance_90",euclidean_distance)

#loading the file
#euclidean_distance=np.load('euclidean_distance_90.npy')

#creating a matrix to store predictions
predictions=np.zeros(shape=(len(testData),4))

#loading the testdata to the matrix.
i=0
for r in testData:
    rLine = r.split(',')
    predictions[i][0] = int(rLine[USERIDX])
    predictions[i][1]= int(rLine[MOVIEIDX])
    predictions[i][2] = float(rLine[RATINGIDX])
    i=i+1

#predicting the ratings for every row.
for j in range(len(predictions)):
    u=int(predictions[j][0])
    m=int(predictions[j][1])
    w=[]
    KNN=[]
    k=int(sys.argv[4])
    for x in range(userIDMax-1):
        if(x!=u and R[x,m]!=0):
            KNN.append([euclidean_distance[x,u],R[x,m]])
    KNN= sorted(KNN,key=lambda X:X[0])
    if(len(KNN)<k):
        k=len(KNN)
    for i in range(k):
            #print(j)
            #print(i)
            if i==0:
                w.append([1])
            else:
                w.append([(KNN[k-1][0]-KNN[i][0])/(KNN[k-1][0]-KNN[0][0])])
                if math.isnan(w[i][0]):
                        w[i][0]=0
    sum_n=0
    sum_d=0
    for i in range(k):
        sum_n= sum_n+(KNN[i][1]*w[i][0])
        sum_d=sum_d+(w[i][0])
    predictions[j][3]='%.1f'%(sum_n/sum_d)
    #float(Decimal(sum_n/sum_d).quantize(Decimal('.0')))    

#writing the array values using pandas.
import pandas as pd
df=pd.DataFrame(predictions,)
df.to_csv(outFileName,index=False,header=False)

stop=timeit.default_timer()
print('%.0f'%((stop-start)/60),"minutes",(stop-start)%60,"seconds")
