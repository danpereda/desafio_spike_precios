import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats



def distplot(train_df, feature_cols, num_rows, num_cols, figsize = (12,12)):

    f, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)
    f.suptitle('Distribution of Features', fontsize=16)

    for index, column in enumerate(train_df[feature_cols].columns):
        i,j = (index // num_cols, index % num_cols)
        g = sns.histplot(data = train_df, x = column, color="m", kde = True, label="Skew : %.2f"%(train_df[column].skew()), ax=axes[i,j])
        g = g.legend(loc="best")


    plt.tight_layout()
    plt.show()

def plot_corr_one_vs_all(df, col_interes, figsize = (8, 12)):
    corr = df.corr()[[col_interes]].sort_values(by=col_interes, ascending=False).round(2)
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='BrBG', annot_kws={"fontsize":14})
    heatmap.set_title(f'Features Correlating with {col_interes}', fontdict={'fontsize':18}, pad=16);
    plt.show()