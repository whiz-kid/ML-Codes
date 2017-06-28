class biCluster:

	def __init__(self,vec,left=None,right=None,distance=0.0,id=None):
		self.vec = vec
		self.left = left
		self.right = right
		self.distance = distance
		self.id = id
    

def readFile(filename):
	lines = [line for line in file(filename)]

	#first line is the column names
	colNames = lines[0].strip().split('\t')[1:]
	rowNames=[]
	data = []
	for line in lines[1:]:
		p=line.strip().split('\t')

		#first column in each row is the row names
		rowNames.append(p[0])

		#the data for this row is the remaining columns
		data.append(float(x) for x in p[1:])

	return rowNames,colNames,data



def pearson(v1,v2):
	sum1=sum(v1)
	sum2=sum(v2)

	sum1Sq=sum([pow(v,2) for v in v1])
	sum2Sq=sum([pow(v,2) for v in v2])

	pSum=sum([v1[i]*v2[i] for i in range(len(v1))])

	num=pSum-(sum1*sum2/len(v1))
	den=sqrt((sum1Sq-pow(sum1,2)/len(v1))*(sum2Sq-pow(sum2,2)/len(v1)))
	if den==0: return 0

	return 1.0-num/den

def hcluster(rows,distance=pearson):
	distances = {}
	




