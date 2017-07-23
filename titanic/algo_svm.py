import numpy as np
import pandas as pd
from sklearn import svm

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

train["Age"]=train["Age"].fillna(train["Age"].median())
train["Fare"]=train["Fare"].fillna(train["Fare"].median())
train["Sex"][train["Sex"]=="female"]=1
train["Sex"][train["Sex"]=="male"]=0

features=train[["Pclass", "Sex", "Age", "Fare"]].values
target=train["Survived"].values

clf=svm.SVC()
clf=clf.fit(features,target)


test["Sex"][test["Sex"]=="female"]=1
test["Sex"][test["Sex"]=="male"]=0
test["Age"]=test["Age"].fillna(test["Age"].median())
test["Fare"]=test["Fare"].fillna(test["Fare"].median())


test_features=test[["Pclass", "Sex", "Age", "Fare"]].values
prediction=clf.predict(test_features)

PassengerId=np.array(test["PassengerId"],int)
solution=pd.DataFrame(prediction,PassengerId,columns=["Survived"])


solution.to_csv("svm_solution.csv",index_label = ["PassengerId"])