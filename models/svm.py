from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils.utils import get_data, scale_data, generate_submission, print_model_performance_metrics

X, y = get_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=57)

X_train, X_test = scale_data(X_train, X_test)

kernels = ["poly", "sigmoid"]
c_values = [1, 3, 5, 7, 10]

for kernel in kernels:
    print("\nSVC kernel=", kernel)
    for c in c_values:
        print("C=", c)
        svc = SVC(C=c, kernel=kernel, probability=True)
        svc.fit(X_train, y_train)
        prediction = svc.predict(X_test)
        print_model_performance_metrics(y_test, prediction)
        generate_submission(svc, f"output_{kernel}_c={c}.csv")
