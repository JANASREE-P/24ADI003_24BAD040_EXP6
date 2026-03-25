import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

print("JANASREE 24BAD040")

df = pd.read_csv("C:\\Users\\janas\\Downloads\\diabetes_bagging.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = pd.concat([X_train, y_train], axis=1)

class_0 = train_df[train_df["Outcome"] == 0]
class_1 = train_df[train_df["Outcome"] == 1]

min_count = min(len(class_0), len(class_1))

class_0_sampled = class_0.sample(min_count, random_state=42)
class_1_sampled = class_1.sample(min_count, random_state=42)

sampled_train_df = pd.concat([class_0_sampled, class_1_sampled]).sample(frac=1, random_state=42)

X_train = sampled_train_df.drop("Outcome", axis=1)
y_train = sampled_train_df["Outcome"]

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)
bag.fit(X_train, y_train)
bag_pred = bag.predict(X_test)
bag_accuracy = accuracy_score(y_test, bag_pred)
bag_precision = precision_score(y_test, bag_pred)

print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Precision:", dt_precision)
print("Bagging Accuracy:", bag_accuracy)
print("Bagging Precision:", bag_precision)

models = ["Decision Tree", "Bagging"]
accuracies = [dt_accuracy, bag_accuracy]

plt.figure(figsize=(6, 4))
plt.bar(models, accuracies)
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)
plt.show()

cm = confusion_matrix(y_test, bag_pred)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j], ha='center', va='center')

plt.colorbar()
plt.show()
