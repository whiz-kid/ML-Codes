from math import *
from data1 import dataset

def distanceEuclid(v1,v2):

	#print([field for field in data[person1]])
	#print([field for field in data[person2]])
	sumOfSquare=sum([pow(v1[i]-v2[i],2) for i in range(len(v1))])
	if(not sumOfSquare):return 0
	return 1-(1/(1+sumOfSquare))

def distancePearson(v1,v2):


	n=len(v1)
	sum1=sum(v1)
	sum2=sum(v2)

	sumS1=sum([pow(v,2)] for v in v1)
	sumS2=sum([pow(v,2)] for v in v2)

	pSum = sum([v1[i]*v2[i] for i in range(n)])

	num=n*pSum - (sum1*sum2)
	den=sqrt((n*sumS1-pow(sum1,2))*(n*sumS2-pow(sum2,2)))

	if den==0: return 0

	return 1.0-num/den


def createFile(data):
	fd=open('data1.txt','w')
	typeOf=[]
	for name in sorted(data):
		typeOf += [field for field in sorted(data[name]) if field not in typeOf]

	for field in typeOf:
		fd.write('\t%s'%field)
	fd.write('\n')

	for name,data in sorted(data.items()):
		fd.write('%s\t'%name)
		for field in sorted(data):
			#bug - what if the field is empty
			fd.write('%f\t'%data[field])
		fd.write('\n')

	return('data1.txt')


def readFile(filename):
	file=open(filename)
	lines = [line for line in file]

	#first line is the column titles
	colnames=lines[0].strip().split('\t')[1:]

	rownames=[]
	data=[]

	for line in lines[1:]:
		p=line.strip().split('\t')
		#first column is the row name
		rownames.append(p[0])
		#remainder is the data for this row
		data.append([float(x) for x in p[1:]])

	return colnames,rownames,data


def rotateMatrix(data):
	newData=[]
	for i in range(len(data[0])):
		newRow=[data[j][i] for j in range(len(data))]
		newData.append(newRow)

	return newData


#class for representing clusters
class Cluster:
	def __init__(self,name,ratings,left=None,right=None,distance=0.0):
		self.name=name
		self.ratings=ratings
		self.left=left
		self.right=right
		#self.distance=distance


#function for hierarichal clustering algotrithm
def hCluster(rows,rownames):

	cluster=[Cluster(rownames[i],rows[i]) for i in range(len(rows))]
	#print(len(rows))
	#print(len(cluster))
	while len(cluster)>1:

		#print([c.name for c in cluster])

		minDis=distanceEuclid(cluster[0].ratings,cluster[1].ratings)
		node1=0
		node2=1

		for i in range(len(cluster)):
			for j in range(i+1,len(cluster)):
				#print(i,j)
				distance=distanceEuclid(cluster[i].ratings,cluster[j].ratings)
				#print(distance)
				if(distance<minDis):
					minDis=distance
					node1=i
					node2=j


		#make cluster of node1 and node2
		c1=cluster[node1]
		c2=cluster[node2]
		newRating=[(c1.ratings[k]+c2.ratings[k])/2 for k in range(len(rows[node1]))]
		name=c1.name+c2.name
		#print(name)
		newCluster=Cluster(name,newRating,c1,c2)
		cluster.append(newCluster)
		#print("Before removing"+str(len(cluster)))
		index=cluster.index(c1)
		cluster.remove(cluster[index])
		index=cluster.index(c2)
		cluster.remove(cluster[index])
		#print("After removing"+str(len(cluster)))

	return cluster

def pCluster(cluster):

	print(cluster.name)
	print(cluster.ratings)

	if(cluster.left):
		pCluster(cluster.left)
	if(cluster.right):
		pCluster(cluster.right)


column,row,data=readFile(createFile(dataset))
cluster=hCluster(data,row)
#print(len(cluster))
#print(cluster[0].right.ratings)
pCluster(cluster[0])