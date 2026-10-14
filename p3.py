import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris_data = load_iris()
x = iris_data.data
y = iris_data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

base_dt = DecisionTreeClassifier(max_depth=3, random_state=42)

bagging_model = BaggingClassifier(estimator=base_dt, n_estimators=50, random_state=42, bootstrap=True)
bagging_model.fit(x_train, y_train)
y_pred_bagging = bagging_model.predict(x_test)

acc = accuracy_score(y_test, y_pred_bagging)
print("Accuracy of Bagging Classifier:", acc)

cm = confusion_matrix(y_test, y_pred_bagging)
print("Confusion Matrix of Bagging Classifier:")
print(cm)

cr = classification_report(y_test, y_pred_bagging, target_names=iris_data.target_names)
print("Classification Report of Bagging Classifier:")
print(cr)

boost_model = AdaBoostClassifier(estimator=base_dt, n_estimators=50, learning_rate=0.5, random_state=42)
boost_model.fit(x_train, y_train)
y_pred_boost = boost_model.predict(x_test)

acc = accuracy_score(y_test, y_pred_boost)
cm = confusion_matrix(y_test, y_pred_boost)
cr = classification_report(y_test, y_pred_boost, target_names=iris_data.target_names)


print("Confusion Matrix of AdaBoost Classifier:",cm)
print("Accuracy of AdaBoost Classifier:", acc)
print("Classification Report of AdaBoost Classifier:", cr)