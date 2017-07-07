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
combine=pd.concat([train.drop("Survived",1),test])



combine["Child"]=combine["Age"]<=10
combine["Family"]=combine["Parch"]+combine["SibSp"]
combine["Alone"]=combine["Parch"]+combine["SibSp"]==0
combine["Largefamily"]=(combine["Parch"]>3) | (combine["SibSp"]>2)
combine[""]