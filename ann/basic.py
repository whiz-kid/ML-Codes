import numpy as np
import warnings
warnings.filterwarnings('ignore')
import math
# import pandas as pd
# import matplotlib.pyplot as plt


# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier


def sigmoid(x):
  f = 1 / (1 + math.exp(-x))
  if f>=0:
  	return 1
  else:
  	return 0

def backpropagation(train,test,etah):
	# weight values for 2 output and 1 hidden layers
	w=np.zeros(9)

	# output value for 2 output and 1 hidden layers
	h=np.zeros(3)

	# for each training instance
	for i in range(len(train)):
		x1,x2=train[i,0],train[i,1]
		y=test[i]

		# forward propagation for calculating output for every unit
		j=0
		for i in range(0,9,3):
			h[j]=sigmoid(w[i] + w[i+1]*x1 + w[i+2]*x2)
			# print(h[j])
			j+=1

		# backward propagation for error calculation
		delta=np.zeros(3)
		# for output layer
		delta[2]=h[2]*(1-h[2])*(y-h[2])
		# for hidden layers
		delta[1]=h[1]*(1-h[1])*(w[8]*delta[2])
		delta[0]=h[0]*(1-h[0])*(w[7]*delta[2])

		print(delta[0],delta[1],delta[2])
		x=np.array([1,x1,x2,1,x1,x2,1,x1,x2])
		d=np.array([delta[0],delta[0],delta[0],delta[1],delta[1],delta[1],delta[2],delta[2],delta[2]])
		# updating the network wight
		for i in range(9):
			w[i]=w[i]+etah*d[i]*x[i]



	return w

train=np.array([[0,0],[0,1],[1,0],[1,1]])
test=np.ravel([0,1,1,0])
weights=backpropagation(train,test,0.1)
print(weights)


# x=np.array([[0,0],[0,1],[0,1.5],[1,1],[1,0],[-1,0],[0,-1],[0.5,0.5],[0.6,-0.2],[-0.5,-0.4],[-0.2,1.0],\
# 			[3,3],[-4,4],[5,7],[4,-6],[6,9],[-6,-4],[-4,-6],[9,4],[-9,5],[-9,-5],[4,-4]])
# y=np.ravel([1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0])

# x_test=np.array([[0,0.1],[-1,0.6],[2,2.4],[-2,-2]])

# clf=LogisticRegression()
# eq=clf.fit(x,y)
# w=eq.coef_
# w1,w2=w[0,0],w[0,1]
# w0=float(eq.intercept_)
# plt.plot(kind='scatter',x=x[:,0],y=z[:,1])
# plt.show()























