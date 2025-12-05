import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import missingno as msn
from IPython.display import display
import haversine as hs
from matplotlib.gridspec import GridSpec
from scipy.stats import chi2_contingency, f_oneway,jarque_bera,probplot
from sklearn.preprocessing import PowerTransformer

def numerical_analysis(df:pd.DataFrame,col,cat_col=None,bin="auto"):


    print(df[col].describe())
    fig = plt.figure(figsize=(11,8))
    grid = GridSpec(nrows=2,ncols=2,figure=fig)

    ax1 = fig.add_subplot(grid[0,0])
    ax2 = fig.add_subplot(grid[0,1])
    ax3 = fig.add_subplot(grid[1,0])


    sns.kdeplot(data=df, x=col,fill=True,ax=ax1,color='blue')
    ax1.set_title(f"KDE-plot of {col}",fontdict={"size":11,"fontweight":'bold'})
    sns.boxplot(data=df, x=col,hue=cat_col,fill=True,ax=ax2)
    ax2.set_title(f"Box-plot of {col}",fontdict={"size":11,"fontweight":'bold'})
    sns.histplot(data=df, x=col,fill=True,kde = True,ax=ax3,color='green',bins=bin)
    ax3.set_title(f"Hist-plot of {col}",fontdict={"size":11,"fontweight":'bold'})

    plt.suptitle(f"Numerical Analysis of {col}:",fontsize=14,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()


def numerical_categorical_analysis(df:pd.DataFrame,num_col,cat_col):
    fig,(ax1,ax2) = plt.subplots(2,2,figsize=(15,7.5))
    
    sns.barplot(data=df,x=cat_col,y=num_col,color='green',ax=ax1[0])
    ax1[0].set_title("Barplot bewteen these two columns",fontsize=12,fontweight='bold')

    sns.boxplot(data=df,x=num_col,y=cat_col,ax=ax1[1],hue=cat_col)
    ax1[1].set_title("Boxplot bewteen these two columns",fontsize=12,fontweight='bold')

    sns.violinplot(data=df,x=num_col,y=cat_col,ax=ax2[0],hue=cat_col)
    ax2[0].set_title("violinplot bewteen these two columns",fontsize=12,fontweight='bold')

    sns.stripplot(data=df,x=num_col,y=cat_col,ax=ax2[1],hue=cat_col)
    ax2[1].set_title("stripplot bewteen these two columns",fontsize=12,fontweight='bold')

    plt.suptitle(f"Numerical-Categorical Analysis between {num_col} & {cat_col}:",fontsize=14,fontweight='bold')

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

def categorical_analysis(df:pd.DataFrame,cat_col):
    display(pd.DataFrame({"Count":df[cat_col].value_counts(),
             "Percentage":round((df[cat_col].value_counts(normalize=True))*100,2).astype("str")+"%"}).reset_index())
    print("="*100)
    unique_list = df[cat_col].dropna().unique().tolist()
    print(f"For {cat_col}, these are the unique values:\n{unique_list}\nThere are total {df[cat_col].nunique()} unique values.")
    print("="*100)
    print("\n")
    fig = px.histogram(df,cat_col,text_auto=True,color_discrete_sequence=['#219ebc'])
    fig.show()

def multivariate_analysis(df:pd.DataFrame,num_col,cat_col1,cat_col2):
    fig,(ax1,ax2)=plt.subplots(2,2,figsize=(15,7.5))

    sns.barplot(data=df,x=cat_col1,y=num_col,hue=cat_col2,ax=ax1[0])
    ax1[0].set_title(f"Barplot between {cat_col1} & {num_col}",fontsize=12,fontweight='bold')

    sns.boxplot(data=df,x=cat_col1,y=num_col,hue=cat_col2,gap=0.1,ax=ax1[1])
    ax1[1].set_title(f"Boxplot between {cat_col1} & {num_col}",fontsize=12,fontweight='bold')

    sns.violinplot(data=df,x=cat_col1,y=num_col,hue=cat_col2,dodge=True,ax=ax2[0])
    ax2[0].set_title(f"violinplot between {cat_col1} & {num_col}",fontsize=12,fontweight='bold')

    sns.stripplot(data=df,x=cat_col1,y=num_col,hue=cat_col2,dodge=True,ax=ax2[1])
    ax2[1].set_title(f"violinplot between {cat_col1} & {num_col}",fontsize=12,fontweight='bold')

    plt.suptitle(f"Multivariate Analysis among {cat_col1},{cat_col2} and {num_col}:",fontsize=14,fontweight='bold')
    plt.show()

# test the relationship between two categorical features
def chi2_test(df:pd.DataFrame,col1,col2,alpha=0.05):
    print(f"Null Hypothesis: There is no relationship between {col1} & {col2}.")
    data = df.loc[:,[col1,col2]].dropna()
    contengency_tab = pd.crosstab(data[col1],data[col2])
    _,p_val,_,_=chi2_contingency(contengency_tab)
    print(p_val)
    if p_val<alpha:
        print(f"Reject the null hypothesis.\nThere is a strong relationship between {col1} & {col2}.")
    else:
        print(f"Fail to reject the null hypothesis.\nThere is a no relationship between {col1} & {col2}.")

def ANOVA(df:pd.DataFrame,num_col,cat_col,alpha=0.05):
    print(f"Null Hypothesis: There is no relationship between {num_col} & {cat_col}.")
    data = df.loc[:,[num_col,cat_col]].dropna()

    cat_group = data.groupby(cat_col)
    groups = [group[num_col].values for _,group in cat_group]
    f_stat,p_val =f_oneway(*groups)
    print(p_val)
    
    if p_val<alpha:
        print(f"Reject the null hypothesis.\nThere is a strong relationship between {num_col} & {cat_col}.")
    else:
        print(f"Fail to reject the null hypothesis.\nThere is a no relationship between {num_col} & {cat_col}.")

def test_for_normality(df,col,alpha=0.05):
    print(f"Null Hypothesis: {col} is normally distributed.")
    data = df[col]
    _,p_val = jarque_bera(data)
    print(p_val)
    if p_val<alpha:
        print(f"Reject the null hypothesis.\n{col} is not normally distriuted.")
    else:
        print(f"Fail to reject the null hypothesis.\n{col} is normally distriuted.")




def impute_analysis(df,col_name,strategy ='mode',val=None):
    if df[col_name].dtype == 'object':
        fig = make_subplots(rows=1,cols=2,subplot_titles=['Before Imputation','After Imputation'])


        fig.add_trace(go.Histogram(x=df[col_name]),row=1,col=1)

        if strategy == 'mode':
            fig.add_trace(go.Histogram(x=df[col_name].fillna(df[col_name].mode()[0]),marker={'color':"green"}),row=1,col=2)
        elif strategy == 'constant':
            fig.add_trace(go.Histogram(x=df[col_name].fillna(val),marker={'color':"green"}),row=1,col=2)


        fig.update_layout(title = {"text":f"{col_name} column Imputation Analysis with {strategy}:","x":0.5,"xanchor":"center"},font = {"family":"arial black"},width = 1000, height = 500)

        fig.show()
    
    elif (df[col_name].dtype == 'int') or (df[col_name].dtype == 'float'):
        plt.figure(figsize=(12, 5))

        # --- Subplot 1: Before Imputation ---
        plt.subplot(1, 2, 1)
        sns.kdeplot(df[col_name], fill=True)
        plt.title("Before Imputation")

        # --- Subplot 2: After Imputation ---
        plt.subplot(1, 2, 2)

        if strategy == 'mean':
            sns.kdeplot(df[col_name].fillna(df[col_name].mean()), fill=True,color='green')
            plt.title("After Imputation (Mean)")
        elif strategy == 'mode':
            sns.kdeplot(df[col_name].fillna(df[col_name].mode()[0]), fill=True,color='green')
            plt.title("After Imputation (Mode)")
        elif (strategy=='constant') and (val is not object):
            sns.kdeplot(df[col_name].fillna(val), fill=True,color='green')
            plt.title("After Imputation (Constant)")

        plt.tight_layout()
        plt.show()