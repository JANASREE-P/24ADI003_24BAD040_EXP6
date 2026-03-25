print("JANASREE 24BAD040")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("C:\\Users\\janas\\Downloads\\churn_boosting.csv")

le = LabelEncoder()
df['ContractType'] = le.fit_transform(df['ContractType'])
df['InternetService'] = le.fit_transform(df['InternetService'])
df['Churn'] = le.fit_transform(df['Churn'])

X = df[['Tenure', 'MonthlyCharges', 'ContractType', 'InternetService']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = pd.concat([X_train, y_train], axis=1)

class_0 = train_df[train_df['Churn'] == 0]
class_1 = train_df[train_df['Churn'] == 1]

min_count = min(len(class_0), len(class_1))

class_0_sampled = class_0.sample(min_count, random_state=42)
class_1_sampled = class_1.sample(min_count, random_state=42)

sampled_train_df = pd.concat([class_0_sampled, class_1_sampled]).sample(frac=1, random_state=42)

X_train = sampled_train_df.drop('Churn', axis=1)
y_train = sampled_train_df['Churn']

ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)
ada_model.fit(X_train, y_train)

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)

y_pred_ada = ada_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

ada_accuracy = accuracy_score(y_test, y_pred_ada)
ada_precision = precision_score(y_test, y_pred_ada)

gb_accuracy = accuracy_score(y_test, y_pred_gb)
gb_precision = precision_score(y_test, y_pred_gb)

y_prob_ada = ada_model.predict_proba(X_test)[:, 1]
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_prob_ada)
auc_ada = auc(fpr_ada, tpr_ada)

y_prob_gb = gb_model.predict_proba(X_test)[:, 1]
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)
auc_gb = auc(fpr_gb, tpr_gb)

print("AdaBoost Accuracy:", ada_accuracy)
print("AdaBoost Precision:", ada_precision)
print("AdaBoost AUC:", auc_ada)

print("Gradient Boosting Accuracy:", gb_accuracy)
print("Gradient Boosting Precision:", gb_precision)
print("Gradient Boosting AUC:", auc_gb)

plt.figure()
plt.plot(fpr_ada, tpr_ada, label="AdaBoost")
plt.plot(fpr_gb, tpr_gb, label="Gradient Boosting")
plt.plot([0, 1], [0, 1], '--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

features = X.columns

plt.figure()
plt.bar(features, ada_model.feature_importances_)
plt.title("Feature Importance - AdaBoost")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

plt.figure()
plt.bar(features, gb_model.feature_importances_)
plt.title("Feature Importance - Gradient Boosting")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
