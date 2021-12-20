from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from feature_selection.filter_method import remove_collinear_features
from utils.utils import get_data, scale_data, generate_submission, print_model_performance_metrics

X, y = get_data()

to_drop = remove_collinear_features(X, 0.60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=57)

X_train, X_test = scale_data(X_train, X_test)

clf = SGDClassifier(loss="perceptron", max_iter=10000, random_state=5)
params = {
    'penalty': ['None', 'l2', 'l1', 'elasticnet'],
    'alpha': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    'shuffle': [False, True],
    'eta0': [0.01, 0.1, 0.5, 1, 1.5]
}
grid_search = GridSearchCV(clf, param_grid=params, scoring='f1_micro')

grid_search.fit(X_train, y_train)
print("Best params: {}".format(grid_search.best_params_))
print("Best f1 score: %.5f" % grid_search.best_score_)

prediction = grid_search.best_estimator_.predict(X_test)

print_model_performance_metrics(y_test, prediction)

generate_submission(grid_search.best_estimator_, "output.csv", to_drop)
