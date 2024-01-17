# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:05:45 2024

@author: amitvikram.dutta
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import seaborn as sns
import os


#os.chdir(r'C:/Users/Amitvikram.Dutta/Documents/Python Scripts')

df = pd.read_csv(r'./bcwo.data',sep=',',header=None)
df.columns = ['Sample_No','Clump_Thickness','U_Cell_Size','U_Cell_Shape','Marginal_Adhesion','Single_cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses','Class']

df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'], errors='coerce')
df = df.dropna()
#df['Bare_Nuclei'] = df['Bare_Nuclei'].astype(int)  
#df = df.astype({'Bare_Nuclei':'float'})
df['Class'] = df['Class'].replace({2: 0, 4: 1})

train_df = df.sample(frac=0.8,random_state=200)
test_df = df.drop(train_df.index)


logreg = LogisticRegression()
x_train = train_df[['Clump_Thickness','U_Cell_Size','U_Cell_Shape','Marginal_Adhesion','Single_cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses']]
x_test = test_df[['Clump_Thickness','U_Cell_Size','U_Cell_Shape','Marginal_Adhesion','Single_cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses']]

y_train = train_df['Class']
y_test = test_df['Class']

logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print("Accuracy:", logreg.score(x_test, y_test))
# plt.scatter(x_test,y_test)
# plt.plot(x_test,y_pred)

# # # Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

