import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

print("JANASREE 24BAD040")

df = pd.read_csv("C:\\Users\\janas\\Downloads\\fraud_smote.csv")

print(df.head())

print("\nClass Distribution Before SMOTE:")
print(df["Fraud"].value_counts())

df["Fraud"].value_counts().plot(kind='bar')
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

X = df.drop("Fraud", axis=1)
y = df["Fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model_before = LogisticRegression(max_iter=1000)
model_before.fit(X_train, y_train)

y_prob_before = model_before.predict_proba(X_test)[:, 1]
precision_before, recall_before, _ = precision_recall_curve(y_test, y_prob_before)
auc_before = auc(recall_before, precision_before)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nClass Distribution After SMOTE:")
print(pd.Series(y_train_smote).value_counts())

pd.Series(y_train_smote).value_counts().plot(kind='bar')
plt.title("Class Distribution After SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

model_after = LogisticRegression(max_iter=1000)
model_after.fit(X_train_smote, y_train_smote)

y_prob_after = model_after.predict_proba(X_test)[:, 1]
precision_after, recall_after, _ = precision_recall_curve(y_test, y_prob_after)
auc_after = auc(recall_after, precision_after)

print("\nAUC Before SMOTE:", auc_before)
print("AUC After SMOTE:", auc_after)

plt.figure()
plt.plot(recall_before, precision_before, label="Before SMOTE")
plt.plot(recall_after, precision_after, label="After SMOTE")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()
