# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:05:45 2024

@author: amitvikram.dutta
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import os


#os.chdir(r'C:/Users/Amitvikram.Dutta/Documents/Python Scripts')

df = pd.read_csv(r'./bcwo.data',sep=',',header=None)
df.columns = ['Sample_No','Clump_Thickness','U_Cell_Size','U_Cell_Shape','Marginal_Adhesion','Single_cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses','Class']

df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'], errors='coerce')
df = df.dropna()
#df['Bare_Nuclei'] = df['Bare_Nuclei'].astype(int)  
#df = df.astype({'Bare_Nuclei':'float'})

train_df = df.sample(frac=0.8,random_state=200)
test_df = df.drop(train_df.index)


knn = KNeighborsClassifier()
x_train = train_df[['Clump_Thickness','U_Cell_Size','U_Cell_Shape','Marginal_Adhesion','Single_cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses']]
x_test = test_df[['Clump_Thickness','U_Cell_Size','U_Cell_Shape','Marginal_Adhesion','Single_cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses']]

y_train = train_df['Class']
y_test = test_df['Class']

fit = knn.fit(x_train,y_train)
score = knn.score(x_test,y_test)

