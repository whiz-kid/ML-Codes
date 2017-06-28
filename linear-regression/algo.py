import random
import matplotlib.pyplot as plt
import numpy as np

def writeFile():
	file=open('data.txt','w')

	#No of features
	#n=random.randint(1,10)
	n=2

	#Generate the coefficients of X[i]
	O=[]
	for i in range(n):
		O.append(random.randint(1,10))
	print(O)


	#Generate the Equation
	h=''
	for i in range(n):
		h+= str(O[i])+"*x"+str(i)
		if(i!=(n-1)):
			h+= " + "
	#print(h)


	#Write the column names in the file
	for i in range(n):
		file.write("x%d\t" %i)
	file.write('y')
	file.write('\n')


	#No of training set
	#m=random.randint(50,100)
	m=20

	y=[]
	#Generate the training Set
	for i in range(m):
		y.append(1*O[0])
		file.write("1\t")
		for j in range(1,n):
			x=random.randint(0,10)
			y[i]+=x*O[j]
			file.write('%d\t' %x)
		file.write('%d'%y[i])
		file.write('\n')


	return n,m,O

def readFile(filename):
	file=open(filename)

	lines=[line for line in file]
	colnames=lines[0].strip().split('\t')
	#print(colnames)
	noOfFeatures=len(colnames)-1

	x=[]
	y=[]

	for line in lines[1:]:
		p=line.strip().split('\t')
		for i in range(1,len(p)-1):
			x.append(p[i])
		y.append(p[i+1])

	x=map(int,x)
	y=map(int,y)

	return x,y


def gradientDescent(x,y):
	
	m=len(x)
	O=[]
	temp=[]
	#for i in range(# dimension of x):
	#	O.append(0)

	#How to find iimension of List
	O.append(0)
	O.append(0)
	temp.append(0)
	temp.append(0)
	a=0.1

	while True:
		temp[0]=O[0]-(a*sum([(O[0]+O[1]*x[i]-y[i]) for i in range(m)]))/m
		temp[1]=O[1]-(a*sum([(O[0]+O[1]*x[i]-y[i])*x[i] for i in range(m)]))/m
		print(temp)
		if O[0]==temp[0] and O[1]==temp[1]:
			break
		O[0]=temp[0]
		O[1]=temp[1]

	fig=plt.figure()
	ax1=fig.add_subplot(331)
	ax1.scatter(x,y)

	ax2=fig.add_subplot(332)
	ax2.plot()

	plt.show()

	print(O)



#n,m,O=writeFile()
x,y=readFile('data.txt')
gradientDescent(x,y)

