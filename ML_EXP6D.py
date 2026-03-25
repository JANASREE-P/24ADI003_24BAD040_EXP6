import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

print("JANASREE 24BAD040")

df = pd.read_csv("C:\\Users\\janas\\Downloads\\heart_stacking.csv")

X = df[['Age', 'Cholesterol', 'MaxHeartRate', 'RestingBP']]
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

train_df = pd.concat([X_train, y_train], axis=1)

class_0 = train_df[train_df['HeartDisease'] == 0]
class_1 = train_df[train_df['HeartDisease'] == 1]

min_count = min(len(class_0), len(class_1))

class_0_sampled = class_0.sample(min_count, random_state=42)
class_1_sampled = class_1.sample(min_count, random_state=42)

sampled_train_df = pd.concat([class_0_sampled, class_1_sampled]).sample(frac=1, random_state=42)

X_train = sampled_train_df.drop('HeartDisease', axis=1)
y_train = sampled_train_df['HeartDisease']

X_train = X_train + np.random.normal(0, 0.5, X_train.shape)

lr = LogisticRegression(max_iter=1000)
svm = SVC(probability=True, C=0.5)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)

lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
dt.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_dt = dt.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
acc_svm = accuracy_score(y_test, y_pred_svm)
acc_dt = accuracy_score(y_test, y_pred_dt)

print("Logistic Regression Accuracy:", acc_lr)
print("SVM Accuracy:", acc_svm)
print("Decision Tree Accuracy:", acc_dt)

estimators = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('svm', SVC(probability=True, C=0.5)),
    ('dt', DecisionTreeClassifier(max_depth=3, random_state=42))
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

stack_model.fit(X_train, y_train)

y_pred_stack = stack_model.predict(X_test)
acc_stack = accuracy_score(y_test, y_pred_stack)

print("Stacking Accuracy:", acc_stack)

models = ['Logistic Regression', 'SVM', 'Decision Tree', 'Stacking']
accuracies = [acc_lr, acc_svm, acc_dt, acc_stack]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()
