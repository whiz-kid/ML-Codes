import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import csv

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def predict(train,w):
	y=w[0]
	for i in range(0,len(train)-1):
		y += train[i]*w[i+1]

	return sigmoid(y)


def algo_logistic(train,a,epoch):
	#length of csv file is showing one greater
	m=len(train)
	n=len(train[0])-1
	w=[0.0 for i in range(n+1)]
	for e in range(epoch):
	#while True:
		M=[0.0 for i in range(n+1)]
		for i in range(m):
			h=predict(train[i],w)
			#print(h)
			M[0] += (h-train[i][-1])
			for j in range(1,n+1):
				M[j] += (h-train[i][-1])*train[i][j-1]
			#m2 += (h-points[i,2])*points[i,1]


		#temp=[0.0 for i in range(n)]
		for i in range(n+1):
			w[i]=w[i]-(a*M[i]/m)

	return w


def algo_logistic_stochastic(train,a,epoch):
	m=len(train)
	n=len(train[0])-1
	w=[0.0 for i in range(n+1)]
	for e in range(epoch):
		M=[0.0 for i in range(n+1)]
		for i in range(m):
			h=predict(train[i],w)
			#print(h)
			M[0] = (h-train[i][-1])
			for j in range(1,n+1):
				M[j] = (h-train[i][-1])*train[i][j-1]

			for i in range(n+1):
				w[i]=w[i]-(a*M[i]/m)

	return w

train=[]
with open('train.csv') as csvFile:
	readCsv=csv.reader(csvFile,delimiter=',')
	for row in readCsv:
		row=list(map(float,row))
		train.append(row)
w=algo_logistic_stochastic(train,0.01,100)
#print(w)
test=[]
with open('test.csv') as csvFile:
	readCsv=csv.reader(csvFile,delimiter=',')
	for row in readCsv:
		row=list(map(float,row))
		test.append(row)

result=[]

for i in range(len(test)):
	res=predict(test[i],w)
	if(res>=0.5):
		res=1.0
	else:
		res=0.0

	result.append([test[i][-1],res])


with open('result.csv','w') as csvFile:
	csvWriter=csv.writer(csvFile)
	csvWriter.writerows(result)


j=0
for i in range(len(result)):
	if result[i][0]==result[i][1]:
		j+=1

print(j/float(i)*100)