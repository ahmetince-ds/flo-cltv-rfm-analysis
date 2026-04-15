
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


def grab_col_names(dataframe,cat_th=10,car_th=20):
   
    """
    Veri setindeki değişkenleri kategorik, numerik ve kardinal olarak ayirir.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Değişken isimleri alinmak istenen dataframe
    cat_th : int, optional
        Numerik fakat kategorik olan değişkenler için eşik değeri
    car_th : int, optional
        Kategorik fakat kardinal değişkenler için eşik değeri

    Returns
    -------
    cat_cols : list
        Kategorik değişken listesi
    num_cols : list
        Numerik değişken listesi
    cat_but_car : list
        Kategorik görünümlü kardinal değişken listesi
    """


    cat_cols=[col for col in dataframe.columns if str(dataframe[col].dtypes) in ['category','object','bool']]
    num_but_cat =[col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ['int64','float64']]
    cat_but_car=[col for col in dataframe.columns if dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ['category','object']]

    cat_cols = cat_cols + num_but_cat
    cat_cols =[col for col in cat_cols if col not in cat_but_car]
    num_cols=[col for col in dataframe.columns if dataframe[col].dtypes in ['int64', 'float64']]
    num_cols=[col for col in num_cols if col not in cat_cols]

    print(f'Observations : {dataframe.shape[0]}')
    print(f'Variables : {dataframe.shape[1]}')
    print(f'cat_cols : {len(cat_cols)}')
    print(f'num_cols : {len(num_cols)}')
    print(f'cat_but_car : {len(cat_but_car)}')
    print(f'num_but_cat : {len(num_but_cat)}')


    return cat_cols , num_cols , cat_but_car

def cat_summary(dataframe,col_name,plot=False):
    if dataframe[col_name].dtypes == 'bool':
        dataframe[col_name] = dataframe[col_name].astype(int)
        print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        'Ratio':100*dataframe[col_name].value_counts() / len(dataframe)}))
        if plot:
            sns.countplot(x=dataframe[col_name],data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        'Ratio':100*dataframe[col_name].value_counts() / len(dataframe)}))
        print('#*#*#*#*#*#*#*#*#*#*#*#*#')

        if plot:
            sns.countplot(x=dataframe[col_name],data=dataframe)
            plt.show(block=True)

def num_summary(dataframe,numerical_col,plot=False):
    quantiles = [0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

