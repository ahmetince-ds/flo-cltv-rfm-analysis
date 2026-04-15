import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from eda import grab_col_names, cat_summary, num_summary

df = sns.load_dataset("iris")
print(df.head())

cat_summary,num_summary,grab_col_names(df)

# Korelasyon
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

# Pairplot (efsane grafik) # bu harika bir grafik
sns.pairplot(df, hue="species")
plt.show()

