import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn import tree

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
#print(train["Survived"][train["Sex"]=="female"].value_counts(normalize=True))
#print(train["Survived"][train["Sex"]=="male"].value_counts(normalize=True))


#train["Child"]=float("NaN")
#train["Child"][train["Age"]<=18]=1
#train["Child"][train["Age"]>18]=0
#print(train["Survived"][train["Child"]==1].value_counts())

"""test_one=test
test_one["Survived"]=0
test_one["Survived"][test_one["Sex"]=="female"]=1
test_one["Survived"][test_one["Sex"]=="male"]=0"""

#print(test_one["Survived"])

train["Age"]=train["Age"].fillna(train["Age"].median())
train["Fare"]=train["Fare"].fillna(train["Fare"].median())

train["Sex"][train["Sex"]=="female"]=1
train["Sex"][train["Sex"]=="male"]=0

#train["Child"]=train["SibSp"].values + train["Parch"].values + 1

#print(train)

target=train["Survived"].values
features=train[["Pclass", "Sex", "Age", "Fare"]].values

my_tree=tree.DecisionTreeClassifier(max_depth=10,min_samples_split=5,random_state=1)
my_tree=my_tree.fit(features,target)

#print(my_tree.feature_importances_)
#print(my_tree.score(features,target))

test["Sex"][test["Sex"]=="female"]=1
test["Sex"][test["Sex"]=="male"]=0
test["Age"]=test["Age"].fillna(test["Age"].median())
test["Fare"]=test["Fare"].fillna(test["Fare"].median())

#test["Child"]=test["SibSp"].values + test["Parch"].values + 1

test_features=test[["Pclass", "Sex", "Age", "Fare"]].values
prediction=my_tree.predict(test_features)

PassengerId=np.array(test["PassengerId"],int)
solution=pd.DataFrame(prediction,PassengerId,columns=["Survived"])

solution.to_csv("tree_solution.csv",index_label = ["PassengerId"])

#print(train.isnull().sum())
#print(test.info())
