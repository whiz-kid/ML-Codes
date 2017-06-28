from math import *
from data1 import dataset

def similarityMeasureDistance(data,person1,person2):

	#print([field for field in data[person1]])
	#print([field for field in data[person2]])
	sumOfSquare = sum([pow(data[person1][field]-data[person2][field],2) for field in data[person1] if field in data[person2]])
	if(not sumOfSquare):return 0
	return 1/(1+sumOfSquare)

def similarityMeasurePearson(data,person1,person2):

	n=len(data[person1])
	sum1 = sum([data[person1][field] for field in data[person1] if field in data[person2]])
	sum2 = sum([data[person2][field] for field in data[person2] if field in data[person1]])

	#print(sum1,sum2)

	sumS1 = sum([pow(data[person1][field],2) for field in data[person1] if field in data[person2]])
	sumS2 = sum([pow(data[person2][field],2) for field in data[person2] if field in data[person1]])

	#print(sumS1)
	#print(sumS2)
	psum = sum([data[person1][field] * data[person2][field] for field in data[person1] if field in data[person2]])

	num = n*psum - (sum1*sum2)
	den = sqrt( (n*sumS1-pow(sum1,2)) * (n*sumS2-pow(sum2,2)) )
	#print((n*sumS1-pow(sum1,2)) * (n*sumS2-pow(sum2,2)))
	if(den == 0):
		return 0
	return num/den


def topMatches(data,person):
	scores = [(similarityMeasurePearson(data,person,other),other) for other in data if other != person]
	scores.sort()
	scores.reverse()
	return scores

def getRecommendation(data,person):

	totals = {}
	siSums = {}

	for other in data:
		if other != person:
			sim = similarityMeasurePearson(data,person,other)

			if sim <= 0: continue

			for item in data[other]:
				if item not in data[person] or data[person][item] == 0:
					totals.setdefault(item,0)
					totals[item] += data[other][item]*sim
					siSums.setdefault(item,0)
					siSums[item] += sim


	ranking = [(total/siSums[item],item) for item,total in totals.items()]
	ranking.sort()
	ranking.reverse()
	return ranking

person = input("Enter the person name: ")
#person2 = input("Enter the second person name: ")
print(topMatches(dataset,person))
