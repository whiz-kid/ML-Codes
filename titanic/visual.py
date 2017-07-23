import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats


"""

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
combine=pd.concat([train.drop("Survived",1),test])

s=train[train["Survived"]==1]
ns=train[train["Survived"]==0]

s_col="blue"
ns_col="red"

plt.figure()
plt.subplot(331)
sns.distplot(s["Age"].dropna().values,bins=range(0,81,1),kde=False,color=s_col)
sns.distplot(ns["Age"].dropna().values,bins=range(0,81,1),kde=False,color=ns_col,axlabel="Age")
plt.subplot(332)
sns.barplot('Sex','Survived',data=train)
plt.subplot(333)
sns.barplot('Pclass','Survived',data=train)
plt.subplot(334)
sns.barplot('Embarked', 'Survived', data=train)
plt.subplot(335)
sns.barplot('SibSp', 'Survived', data=train)
plt.subplot(336)
sns.barplot('Parch', 'Survived', data=train)
plt.subplot(337)
sns.distplot(s["Fare"].dropna().values+1,bins=range(0,81,1),kde=False,color=s_col)
sns.distplot(ns["Fare"].dropna().values+1,bins=range(0,81,1),kde=False,color=ns_col,axlabel='Fare')
plt.subplots_adjust(hspace=0.35,wspace=0.25)
plt.show()

plt.figure()
foo=sns.heatmap(train.drop('PassengerId',axis=1).corr(),vmax=0.6,square=True,annot=True)
plt.show()


msurv=train[(train["Survived"]==1) & (train["Sex"]=='male')]
mnsurv=train[(train["Survived"]==0) & (train["Sex"]=='male')]
fmsurv=train[(train["Survived"]==1) & (train["Sex"]=='female')]
fmnsurv=train[(train["Survived"]==0) & (train["Sex"]=='female')]


plt.figure()
plt.subplot(121)
sns.distplot(fmsurv["Age"].dropna().values,bins=range(1,81,1),kde=False,color=s_col)
sns.distplot(fmnsurv["Age"].dropna().values,bins=range(1,81,1),kde=False,color=ns_col,axlabel="Female Age")
plt.subplot(122)
sns.distplot(msurv["Age"].dropna().values,bins=range(1,81,1),kde=False,color=s_col)
sns.distplot(mnsurv["Age"].dropna().values,bins=range(1,81,1),kde=False,color=ns_col,axlabel="Male Age")
plt.show()

sns.boxplot(x="Pclass",y="Age",hue="Survived",data=train)
sns.violinplot(x="Pclass",y="Age",hue="Survived",data=train,split=True)
plt.show()

sns.factorplot(x="Pclass",y="Survived",hue="Sex",data=train,col="Embarked")
plt.show()

sns.factorplot(x="Pclass",y="Survived",hue="Embarked",data=train,col="Sex")
plt.show()

tab=pd.crosstab(combine["Embarked"],combine['Pclass'])
print(tab)
dummy=tab.div(tab.sum(1).astype(float),axis=0).plot(kind="bar",stacked=False)
dummy=plt.xlabel("Port Embarked")
dummy=plt.ylabel("Pclass")
plt.show(dummy)

sns.barplot(x="Embarked",y="Survived",hue="Pclass",data=train)
plt.show()

tab=pd.crosstab(combine["Embarked"],combine["Sex"])
print(tab)
tab.div(tab.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
plt.xlabel("Port Embarked")
plt.ylabel("Sex")
plt.show()

tab=pd.crosstab(combine["Pclass"],combine["Sex"])
print(tab)
tab.div(tab.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
plt.xlabel("Port Embarked")
plt.ylabel("Sex")
plt.show()

tab=pd.crosstab(train["SibSp"],train["Sex"])
print(tab)
dummy1=tab.div(tab.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
dummy1=plt.xlabel("Siblings")
dummy1=plt.ylabel("Percentage")


tab=pd.crosstab(train["Parch"],train["Sex"])
print(tab)
dummy2=tab.div(tab.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
dummy2=plt.xlabel("Parents")
dummy2=plt.ylabel("Percentag")

plt.show(dummy1)

sns.violinplot(x="Embarked",y="Age",hue="Survived",data=train,split=True)
plt.hlines([0,10], xmin=-1, xmax=3, linestyles="dotted")
plt.show()


plt.figure()
plt.subplot(311)
ax1=sns.distplot(np.log10(s["Fare"][s["Pclass"]==1].dropna().values+1),kde=False,color=s_col)
ax1=sns.distplot(np.log10(ns["Fare"][ns["Pclass"]==1].dropna().values+1),kde=False,color=ns_col,axlabel="Fare")
ax1.set_xlim(0,np.max(np.log10(s["Fare"].dropna().values+1)))

plt.subplot(312)
ax2=sns.distplot(np.log10(s["Fare"][s["Pclass"]==2].dropna().values+1),kde=False,color=s_col)
ax2=sns.distplot(np.log10(ns["Fare"][ns["Pclass"]==2].dropna().values+1),kde=False,color=ns_col,axlabel="Fare")
ax2.set_xlim(0,np.max(np.log10(s["Fare"].dropna().values+1)))

plt.subplot(313)
ax3=sns.distplot(np.log10(s["Fare"][s["Pclass"]==3].dropna().values+1),kde=False,color=s_col)
ax3=sns.distplot(np.log10(ns["Fare"][ns["Pclass"]==3].dropna().values+1),kde=False,color=ns_col,axlabel="Fare")
ax3.set_xlim(0,np.max(np.log10(s["Fare"].dropna().values+1)))

plt.show()

ax=sns.violinplot(x="Pclass",y="Fare",hue="Survived",data=train,split=True)
ax.set_yscale('log')
plt.show()

"""
#print(  combine.where((combine["Pclass"]==1)&(combine["Sex"]=="female"))
#.groupby(["Embarked","Pclass","Sex","Parch","SibSp"]).size()  )


train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
survived=train["Survived"]
combine=pd.concat([train.drop("Survived",1),test])

combine[ "Child"]=combine["Age"]<=10
combine["Family"]=combine["Parch"]+combine["SibSp"]
combine["Alone"]=combine["Parch"]+combine["SibSp"]==0
combine["Largefamily"]=(combine["Parch"]>3) | (combine["SibSp"]>2)
combine["Deck"]=combine["Cabin"].str[0]
combine["Deck"]=combine["Deck"].fillna("U")
combine["Ttype"]=combine["Ticket"].str[0]
combine["Title"]=combine["Name"].str.split(',',expand=True)[1].str.split('.',expand=True)[0]
combine["Young"]=(combine["Age"]<=30)|(combine["Title"].isin(["Master","Miss","Mlle"]))
combine["Shared_Ticket"]=np.where(combine.groupby("Ticket")["Name"].transform('count')>1,1,0)
combine['Ticket_group'] = combine.groupby('Ticket')['Name'].transform('count')

test=combine.iloc[len(train):]
train=combine.iloc[:len(train)]
train["Survived"]=survived
surv = train[train['Survived']==1]
nosurv = train[train['Survived']==0]

#Child
print(pd.crosstab(train["Deck"],train["Survived"]))









#probability distribution
#pmf pdf binomial trial and distribution multinomial trial and distribution
#variance bias tradeoff
#eigen vectors and eigen values
#inner product significance (projection and similarity)
#l2 l1 norms and errors and regularization

#Preprocessing of data: Data normalization
#Model evalution metrices