import csv
from sys import argv

#command line argument
script,file1,file2=argv

list1=[]
with open(file1+".csv") as csvFile:
	reader=csv.reader(csvFile)
	for row in reader:
		list1.append(row[1])

list2=[]
with open(file2+".csv") as csvFile:
	reader=csv.reader(csvFile)
	for row in reader:
		list2.append(row[1])

n=len(list1)
m=0
for i in range(n):
	if(list1[i]!=list2[i]):
		m+=1

print((n-m)/float(n)*100)

