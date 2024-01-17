import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
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

x_train = train_df[['Clump_Thickness','U_Cell_Size','U_Cell_Shape','Marginal_Adhesion','Single_cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses']]
x_test = test_df[['Clump_Thickness','U_Cell_Size','U_Cell_Shape','Marginal_Adhesion','Single_cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses']]

y_train = train_df['Class']
y_test = test_df['Class']

rf = RandomForestClassifier(n_estimators=100, random_state=1)

rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title('Confusion Matrix')
plt.show()
