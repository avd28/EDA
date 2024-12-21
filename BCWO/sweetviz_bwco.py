#import ydata_profiling
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv(r'C:\Users\amitv\Documents\Python_Scripts\EDA\BCWO\bcwo.data',sep=',',header=None)
df.columns = ['Sample_No','Clump_Thickness','U_Cell_Size','U_Cell_Shape','Marginal_Adhesion','Single_cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses','Class']

df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'], errors='coerce')
df = df.dropna()
#df['Bare_Nuclei'] = df['Bare_Nuclei'].astype(int)  
#df = df.astype({'Bare_Nuclei':'float'})
df['Class'] = df['Class'].replace({2: 0, 4: 1})

report = ProfileReport(df, title='bcwo')
report.to_file("bcwo_data.html")