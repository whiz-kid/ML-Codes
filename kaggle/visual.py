import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

s=train[train["Survived"]==1]
ns=train[train["Survived"]==0]

s_col="blue"
ns_col="red"


#Analyzing each features independently
"""plt.figure()
plt.subplot(331)
sns.distplot(s["Age"].dropna(),bins=range(1,81,1),kde=False,color=s_col)
sns.distplot(ns["Age"].dropna(),bins=range(1,81,1),kde=False,color=ns_col,axlabel="Range Of Age")
plt.subplot(332)
sns.barplot(x="Sex",y="Survived",data=train)
plt.subplot(333)
sns.barplot(x="Pclass",y="Survived",data=train)
plt.subplot(334)
sns.barplot(x="Embarked",y="Survived",data=train)
plt.subplot(335)
sns.barplot(x="SibSp",y="Survived",data=train)
plt.subplot(336)
sns.barplot(x="Parch",y="Survived",data=train)
plt.subplot(337)
sns.distplot(s["Fare"].dropna().values+1,bins=range(0,81,1),kde=False,color=s_col)
sns.distplot(ns["Fare"].dropna().values+1,bins=range(0,81,1),kde=False,color=ns_col,axlabel='Fare')
plt.subplots_adjust(hspace=0.4,wspace=0.4)
plt.show()




#Analyzing relation between features
plt.figure()
sns.hotmap(train.drop("PassengerId",axis=1).corr(),vmax=0.6,square=True,annot=True)



#Relation between Age and Sex
msurv=train[(train["Survived"]==1) & (train["Sex"]=='male')]
mnsurv=train[(train["Survived"]==0) & (train["Sex"]=='male')]
fmsurv=train[(train["Survived"]==1) & (train["Sex"]=='female')]
fmnsurv=train[(train["Survived"]==0) & (train["Sex"]=='female')]
plt.figure()
plt.subplot(221)
sns.distplot(msurv["Age"].dropna(),bins=range(1,81,1),kde=False,color=s_col)
plt.subplot(222)
sns.distplot(mnsurv["Age"].dropna(),bins=range(1,81,1),kde=False,color=ns_col)
plt.subplot(223)
sns.distplot(fmsurv["Age"].dropna(),bins=range(1,81,1),kde=False,color=s_col)
plt.subplot(224)
sns.distplot(fmsurv["Age"].dropna(),bins=range(1,81,1),kde=False,color=ns_col)
plt.show()

#Relation between Pclass and Age
#sns.violinplot(x="Pclass",y="Age",hue="Survived",data=train,split=True)
#plt.hlines([0,10], xmin=-1, xmax=3, linestyles="dotted")
#sns.factorplot(x="Pclass",y="Survived",hue="Sex",col="Embarked",data=train)
#sns.factorplot(x="Pclass",y="Survived",hue="Embarked",data=train)
#plt.show()


tab=pd.crosstab(train["Pclass"],train["Sex"])
tab.div(tab.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
plt.show()

"""
