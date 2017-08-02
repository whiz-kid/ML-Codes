import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

import operator
from pandas.tools.plotting import scatter_matrix

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train['Loan_Status']=train['Loan_Status'].astype('category')
train['Loan_Status'].cat.categories=[0,1]

sucess=train.loc[train['Loan_Status']=='Y']
fail=train.loc[train['Loan_Status']=='N']

#print(train.head(1))
cols=['Married','Dependents','Education']
train['Married']=train['Married'].astype('category')
train['Married'].cat.categories=[0,1]
train['Education']=train['Education'].astype('category')
train['Education'].cat.categories=[1,0]
train['Dependents']=train['Dependents'].astype('category')
train['Dependents'].cat.categories=[0,1,2,3]

train['Married']=train['Married'].fillna(train['Married'].mode().iloc[0])
train['Dependents']=train['Dependents'].fillna(train['Dependents'].mode().iloc[0])

X_train=train.loc[:,cols]
y_train=np.ravel(train.loc[:,['Loan_Status']])

clf=DecisionTreeClassifier(criterion='entropy',max_depth=10)
eq=clf.fit(X_train,y_train)

test['Dependents']=test['Dependents'].astype('category')
test['Dependents'].cat.categories=[0,1,2,3]
test['Dependents']=test['Dependents'].fillna(test['Dependents'].mode().iloc[0])
test['Married']=test['Married'].astype('category')
test['Married'].cat.categories=[0,1]
test['Education']=test['Education'].astype('category')
test['Education'].cat.categories=[1,0]

X_test=test.loc[:,cols]
prediction=eq.predict(X_test)
loan_id=np.ravel(test.loc[:,['Loan_ID']])
ans=pd.DataFrame(prediction,loan_id,columns=['Loan_Status'])

ans.loc[ans['Loan_Status']==0,['Loan_Status']]='N'
ans.loc[ans['Loan_Status']==1,['Loan_Status']]='Y'

ans.to_csv('answer.csv',index_label=['Loan_ID'])