from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from ZZ_Practica1.utils import get_data, scale_data, generate_submission, print_model_performance_metrics

X, y = get_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=57)

X_train, X_test = scale_data(X_train, X_test)

svc = SVC(C=1.0, kernel="linear", probability=True)
svc.fit(X_train, y_train)
prediction = svc.predict(X_test)

print_model_performance_metrics(y_test, prediction)

generate_submission(svc, "output.csv")
