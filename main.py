import seaborn as sns
from eda import grab_col_names, cat_summary, num_summary

dataframe = sns.load_dataset('titanic')

cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

for col in cat_cols:
    cat_summary(dataframe, col)

for col in num_cols:
    num_summary(dataframe, col)