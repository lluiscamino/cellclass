# https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
# https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
# https://chrisalbon.com/code/machine_learning/feature_selection/drop_highly_correlated_features/
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame


def print_corr_matrix(X: DataFrame):
    corr = X.corr().abs()
    plt.figure(figsize=(60, 60))
    sns.heatmap(corr, annot=True, cmap=plt.cm.Reds, fmt=".1f")
    plt.show()


# https://stackoverflow.com/a/61938339/8554847
# https://stats.stackexchange.com/questions/175933/why-is-a-correlation-coefficient-threshold-of-r-0-6-among-predictors-commonly/244451
# (best threshold: 0.6)
def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output:
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i + 1):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    print(drops)
    x.drop(columns=drops, inplace=True)

    return drops
