from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from ZZ_Practica1.utils import get_data, scale_data

X, y = get_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=57)

X_train, X_test = scale_data(X_train, X_test)

clf = SGDClassifier(loss="perceptron", eta0=1, max_iter=1000, learning_rate="constant", random_state=5)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

cf_matrix = confusion_matrix(y_test, prediction)
print(cf_matrix)
f1 = f1_score(y_test, prediction, average="micro")
print("F1 score:\t", f1)
