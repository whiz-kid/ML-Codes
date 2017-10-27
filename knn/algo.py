import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import operator


def data():
	arr_trains=np.array([[5.2,2.2,1.0],[5.1,2.4,1.0],[5.8,1.4,0.0],[6.1,1.1,0.0],[5.2,2.3,1.0],[5.9,1.3,0.0]])
	train=pd.DataFrame(arr_trains,columns=['height','width','girl'])
	arr_test=np.array([[5.1,1.8],[6.0,2.0],[5.8,1.9]])
	test=pd.DataFrame(arr_test,columns=['height','width'])
	return train,test


def knn(train,test,k):
	data=np.array(train.iloc[:,0:-1])
	target=np.array(train.iloc[:,-1])
	test=np.array(test.iloc[:])

	y_test=[]
	for point in test:
		rows=data.shape[0]
		diffmat=np.tile(point,(rows,1))-data
		sqdiffmat=diffmat**2
		sqdistance=sqdiffmat.sum(axis=1)
		distance=sqdistance**0.5
		df=np.zeros((rows,2))
		for i in range(len(distance)):
			df[i]=np.array([distance[i],target[i]])
		df=df[np.argsort(df[:,0])]
		#print(df)

		count={1:0,0:0}
		for i in range(k):
			key=df[i,1]
			count[key]=count.setdefault(key,0)+(1/df[i,0])
		count=sorted(count.items(),key=operator.itemgetter(1),reverse=True)
		print(count)
		y_test.append(count[0][0])

	return y_test

train,test=data()
ans=knn(train,test,5)
print(ans)


#lambda function
