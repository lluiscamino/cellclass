# https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
# https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
# https://chrisalbon.com/code/machine_learning/feature_selection/drop_highly_correlated_features/
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame


def print_corr_matrix(X: DataFrame):
    corr = X.corr().abs()
    plt.figure(figsize=(60, 60))
    sns.heatmap(corr, annot=True, cmap=plt.cm.Reds, fmt=".1f")
    plt.show()


def drop_correlated_features(X: DataFrame):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X.drop(to_drop, axis=1, inplace=True)
