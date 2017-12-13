#!/usr/bin/python3

## arg1 -- rating file (training set)
## arg2 -- ratings to predict (test set)
## arg3 -- predictions file (output)



import timeit
start =timeit.default_timer()
import sys
import numpy as np
import math
import csv
from decimal import Decimal, ROUND_UP
import statistics


## Offsets in ratings file for data
USERIDX=0
MOVIEIDX=1
RATINGIDX=2

if len(sys.argv) != 3:
	print('usage: trainingDataFileName testDataFileName predictionsFile')
	sys.exit(99)

trainingFileName = sys.argv[1]
testFileName     = sys.argv[2]
outFileName      = sys.argv[3]



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

R = np.zeros(shape=(movieIDMax,userIDMax))

## build ratings matrix
for r in trainingData:
        rLine = r.split(',')
        userID = int(rLine[USERIDX])
        movieID= int(rLine[MOVIEIDX])
        rating = float(rLine[RATINGIDX])
        R[movieID][userID]  = float(rating)

## If you want to debug this process, uncomment
## the line below so you can see how it is processed
##	print('userid:' + str(userID) + ' movieID:' + 
##          str(movieID) + ' rating:' + str(R[movieID][userID]))

sim_measure=np.full((movieIDMax,movieIDMax),-2,dtype='f')
mean_users=np.full((1,userIDMax),-10,dtype='f')

for u in range(userIDMax):
        mean_users[0,u]=sum(R[:,u])/np.count_nonzero(R[:,u])

def simularity_measure(i,j):
        U=[]
        num=0
        d1=0
        d2=0
        for z in range(userIDMax):
                if R[i][z]!=0 and R[j][z]!=0:
                        u=z
                        mean_u=0
                        k=0
                        mean_u=mean_users[0,u]
                        num=num+((R[i][u]-mean_u)*(R[j][u]-mean_u))
                        d1 = d1+((R[i][u]-mean_u)*(R[i][u]-mean_u))
                        d2 = d2+((R[j][u]-mean_u)*(R[j][u]-mean_u))
        if num==0 or d1==0 or d2==0:
                return 0
        else:
                return (num/(math.sqrt(d1)*math.sqrt(d2)))
print(simularity_measure(0,1))

predictions=np.zeros(shape=(len(testData),4))

i=0
for r in testData:
        rLine = r.split(',')
        predictions[i][0] = int(rLine[USERIDX])
        predictions[i][1]= int(rLine[MOVIEIDX])
        predictions[i][2] = float(rLine[RATINGIDX])
        i=i+1

for l in range(len(predictions)):
        print(l)
        u_id=int(predictions[l][0])
        m_id=int(predictions[l][1])
        movies=[]
        r=[]
        for m in range(movieIDMax):
                if R[m][u_id]!=0:
                        if sim_measure[m][m_id]!=-2:
                                r.append([R[m][u_id],sim_measure[m][m_id]])
                        else:
                                sim_measure[m][m_id]=simularity_measure(m,m_id)
                                sim_measure[m_id][m]=sim_measure[m][m_id]
                                r.append([R[m][u_id],sim_measure[m][m_id]])
        num=0
        den=0
        for n in r:
                if n[1]>0:
                        num=num+(n[0]*n[1])
                        den=den+n[1]
        if num>0 and den>0:
                predictions[l][3]=(Decimal(num/den).quantize(Decimal('.0')))
        else:
                predictions[l][3]=0

import pandas as pd
df=pd.DataFrame(predictions)
df.to_csv(outFileName,index=False,header=False)

stop=timeit.default_timer()
print('%.0f'%((stop-start)/60),"minutes",(stop-start)%60,"seconds")


