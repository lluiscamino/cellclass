import string
from typing import Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()


def get_data() -> Tuple[DataFrame, DataFrame]:
    data = pd.read_csv("../data/train.csv")
    data = data.drop(columns=["idx", "path"])

    X = data.loc[:, data.columns != "class"]
    y = data["class"].values
    return X, y


def scale_data(X_train: DataFrame, X_test: DataFrame) -> Tuple[DataFrame, DataFrame]:
    X_train_scaled = std_scaler.fit_transform(X_train)
    X_test_scaled = std_scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def print_model_performance_metrics(y_true: DataFrame, y_pred: DataFrame):
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)

    precision = precision_score(y_true, y_pred, average="micro")
    print("Precision:", precision)
    recall = recall_score(y_true, y_pred, average="micro")
    print("Recall:   ", recall)
    f1 = f1_score(y_true, y_pred, average="micro")
    print("F1 score: ", f1)


def generate_submission(predictor, output_file: string, to_drop=[]):
    X = pd.read_csv("../data/test.csv")
    X.drop(columns=to_drop, inplace=True)
    idx = X["idx"]
    X = X.loc[:, X.columns != "idx"]
    X = std_scaler.transform(X)

    prediction = predictor.predict(X)

    data = pd.DataFrame([], columns=["idx", "class"])
    data["idx"] = idx
    data["class"] = prediction
    data.to_csv(output_file, index=False)
