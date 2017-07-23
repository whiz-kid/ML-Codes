import warnings
warnings.filterwarnings("ignore")
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


iris=datasets.load_iris()

X=iris.data
y=iris.target
colMap={0:"red",1:"blue",2:"yellow"}
cols=list(map(lambda x:colMap.get(x),iris.target))
df=pd.DataFrame(X,columns=iris.feature_names)

#pd.scatter_matrix(df,c=cols,s=150)
#df.hist()

#sns.countplot(x="s_length",data=train)
#sns.jointplot(x="s_length",y="s_width",data=train,size=5,kind="scatter")
#sns.FacetGrid(train,hue="Species",size=5).map(plt.scatter,"s_length","s_width").add_legend()
#sns.heatmap(train.corr(),annot=True)
#plt.show()


"""
models=[]
models.append(('knn',KNeighborsClassifier()))
models.append(('tree',DecisionTreeClassifier()))
models.append(('logistic',LogisticRegression()))
models.append(('svm',SVC()))
models.append(('bayes',GaussianNB()))
models.append(('discr',LinearDiscriminantAnalysis()))


results=[]
names=[]
for name,model in models:
	kfold=KFold(n_splits=10,random_state=1)
	score=cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
	results.append(score.mean())
	names.append(name)
	print("%s: %f %f"%(name,score.mean(),score.std()))

results=pd.DataFrame(results)
results.plot(kind="bar").set_xticklabels(names)
plt.xlabel('Classifers')
plt.ylabel('Accuracy')
plt.show()"""