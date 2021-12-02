from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from feature_selection.filter_method import remove_collinear_features
from utils.utils import get_data, scale_data, generate_submission, print_model_performance_metrics

X, y = get_data()

to_drop = remove_collinear_features(X, 0.60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=57)

X_train, X_test = scale_data(X_train, X_test)

clf = SGDClassifier(loss="perceptron", eta0=1, max_iter=1000, random_state=5)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

print_model_performance_metrics(y_test, prediction)

generate_submission(clf, "output.csv", to_drop)
