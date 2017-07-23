import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm

#import xgboost as xgb
#from mlxtend.classifier import StackingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV



train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
combine = pd.concat([train.drop('Survived',1),test])
survived = train['Survived']
train["Age"]=train["Age"].fillna(train["Age"].median())


train['Embarked'].iloc[61]="C"
train['Embarked'].iloc[829]="C"
test["Fare"].iloc[152]=combine["Fare"][combine["Pclass"]==3].dropna().median()
combine = pd.concat([train.drop('Survived',1),test])

#combine["Age_known"]=combine["Age"].isnull()==False
combine["Cabin_known"]=combine["Cabin"].isnull()==False
combine[ "Child"]=combine["Age"]<=10
combine["Family"]=combine["Parch"]+combine["SibSp"]
combine["Alone"]=combine["Parch"]+combine["SibSp"]==0
combine["Large_family"]=(combine["Parch"]>=3) | (combine["SibSp"]>=2)
combine["Deck"]=combine["Cabin"].str[0]
combine["Deck"]=combine["Deck"].fillna("U")
combine['Title'] = combine['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
combine['Young'] = (combine['Age']<=30) | (combine['Title'].isin(['Master','Miss','Mlle']))
combine['Ttype'] = combine['Ticket'].str[0]
combine["Shared_ticket"]=np.where(combine.groupby('Ticket')['Name'].transform('count')>1,1,0)

combine["Sex"]=combine["Sex"].astype('category')
combine["Sex"].cat.categories=[0,1]
combine["Sex"]=combine["Sex"].astype('int')
combine["Embarked"]=combine["Embarked"].astype('category')
combine["Embarked"].cat.categories=[0,1,2]
combine["Embarked"]=combine["Embarked"].astype('int')
combine["Deck"]=combine["Deck"].astype('category')
combine["Deck"].cat.categories=[0,1,3,4,5,6,7,8,9]
combine["Deck"]=combine["Deck"].astype('int')



test = combine.iloc[len(train):]
train = combine.iloc[:len(train)]
train['Survived'] = survived

training,testing=train_test_split(train,test_size=0.2,random_state=0)
#print("Total sample size = %i; training sample size = %i, testing sample size = %i"\
#     %(train.shape[0],training.shape[0],testing.shape[0]))

cols=["Sex","Pclass","Cabin_known","Large_family","Alone","Parch","SibSp","Young","Alone",\
		"Shared_ticket","Child","Fare","Age"]
X=training.loc[:,cols]
y=np.ravel(training.loc[:,["Survived"]])
X_test=testing.loc[:,cols]
y_test=np.ravel(testing.loc[:,["Survived"]])

#Perceptron
clf=Perceptron(penalty="l2",n_iter=5,eta0=1,class_weight=None)
eq=clf.fit(X,y)
perceptron_score=eq.score(X,y)

#Logistic Regression
clf=LogisticRegression(penalty="l2",class_weight="balanced",max_iter=100,solver='liblinear',C=1)
eq=clf.fit(X,y)
logistic_score=eq.score(X,y)


#KNeighbor Classification
clf=KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto',metric='minkowski')
eq=clf.fit(X,y)
kneighbor_score=eq.score(X,y)

clf=tree.DecisionTreeClassifier(criterion='entropy',max_features=None,class_weight=None)
eq=clf.fit(X,y)
tree_score=eq.score(X,y)
print(pd.DataFrame(list(zip(cols,eq.feature_importances_))).sort_values(by=1,ascending=False))

clf=svm.SVC(C=1,kernel='rbf',degree=3)
eq=clf.fit(X,y)
svc_score=eq.score(X,y)


clf=RandomForestClassifier(criterion='gini',bootstrap=True,max_depth=10,class_weight=None)
eq=clf.fit(X,y)


