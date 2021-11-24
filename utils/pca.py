import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.utils import get_data

X_original, _ = get_data()
X = StandardScaler().fit_transform(X_original)

pca = PCA()
pca.fit(X)
var_ratio = pca.explained_variance_ratio_
cum_var_ratio = np.cumsum(var_ratio)

fig, ax = plt.subplots()
ax.plot(cum_var_ratio)
ax.set(xlabel='Components', ylabel='Cumulative variance')
ax.grid()
plt.show()
