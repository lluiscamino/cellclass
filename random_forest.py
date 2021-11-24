from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ZZ_Practica1.utils import get_data, scale_data, generate_submission, print_model_performance_metrics

X, y = get_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=57)

X_train, X_test = scale_data(X_train, X_test)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

print_model_performance_metrics(y_test, prediction)

generate_submission(clf, "output.csv")
