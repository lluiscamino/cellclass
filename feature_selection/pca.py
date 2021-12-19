import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from utils.utils import get_data, scale_data


def principal_components_analysis(X):
    pca = PCA()
    pca.fit(X)
    var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)

    fig, ax = plt.subplots()
    ax.plot(cum_var_ratio)
    ax.set(xlabel='Components', ylabel='Cumulative variance')
    ax.grid()

    # https://towardsdatascience.com/visualising-the-classification-power-of-data-54f5273f640
    var = pca.explained_variance_[0:10]  # percentage of variance explained
    labels = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']

    plt.figure(figsize=(15, 7))
    plt.bar(labels, var, )
    plt.xlabel('Pricipal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.show()


X, _ = get_data()

# remove_collinear_features(X, 0.60)

X, _ = scale_data(X, X)

principal_components_analysis(X)
