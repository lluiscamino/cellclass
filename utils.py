from typing import Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler


def get_data() -> Tuple[DataFrame, DataFrame]:
    data = pd.read_csv("data/train.csv")
    data = data.drop("idx", 1)
    data = data.drop("path", 1)

    X = data.loc[:, data.columns != "class"]
    y = data["class"].values
    return X, y


def scale_data(X_train: DataFrame, X_test: DataFrame) -> Tuple[DataFrame, DataFrame]:
    std_scaler = StandardScaler()
    X_train_scaled = std_scaler.fit_transform(X_train)
    X_test_scaled = std_scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
