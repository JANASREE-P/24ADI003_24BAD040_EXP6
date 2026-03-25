import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

print("JANASREE 24BAD040")

df = pd.read_csv("C:\\Users\\janas\\Downloads\\income_random_forest.csv")
print(df.head())

le = LabelEncoder()
df['Income'] = le.fit_transform(df['Income'])

X = df[['Age', 'EducationYears', 'HoursPerWeek', 'Experience']]
y = df['Income']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

train_df = pd.concat([X_train, y_train], axis=1)

class_0 = train_df[train_df['Income'] == 0]
class_1 = train_df[train_df['Income'] == 1]

min_count = min(len(class_0), len(class_1))

class_0_sampled = class_0.sample(min_count, random_state=42)
class_1_sampled = class_1.sample(min_count, random_state=42)

sampled_train_df = pd.concat([class_0_sampled, class_1_sampled]).sample(frac=1, random_state=42)

X_train = sampled_train_df.drop('Income', axis=1)
y_train = sampled_train_df['Income']

X_train = X_train + np.random.normal(0, 0.5, X_train.shape)

rf_model = RandomForestClassifier(
    n_estimators=10,
    max_depth=3,
    random_state=42
)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy After Sampling:", accuracy)

tree_range = [10, 20, 50, 100, 150]
accuracies = []

for n in tree_range:
    model = RandomForestClassifier(
        n_estimators=n,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

plt.figure()
plt.plot(tree_range, accuracies, marker='o')
plt.title("Accuracy vs Number of Trees")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.show()

importances = rf_model.feature_importances_
features = X.columns

plt.figure()
plt.bar(features, importances)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
