from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from feature_selection.filter_method import remove_collinear_features
from utils.utils import get_data, scale_data, print_model_performance_metrics, generate_submission

X, y = get_data()

to_drop = remove_collinear_features(X, 0.99)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=57)

X_train, X_test = scale_data(X_train, X_test)

clf = RandomForestClassifier(random_state=0)
parameters = {
    "max_depth": range(1, 602, 10),
    "n_estimators": range(1, 602, 10),
    "max_features": ["auto", "sqrt", "log2"]
}
"""
parameters = {
    "max_depth": range(1, 602, 10),
    "n_estimators": range(1, 602, 10)
}
"""
grid_search_cv = GridSearchCV(clf, parameters, cv=5)
grid_search_cv.fit(X_train, y_train)
best_clf = grid_search_cv.best_estimator_
print("Best estimator: ", best_clf)

best_clf.fit(X_train, y_train)
prediction = best_clf.predict(X_test)

print_model_performance_metrics(y_test, prediction)

generate_submission(best_clf, "output.csv", to_drop)
