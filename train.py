import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("Traffic.csv")

# --------------------------------------------------
# 2. Data Preprocessing
# --------------------------------------------------

df["Hour"] = pd.to_datetime(
    df["Time"],
    format="%I:%M:%S %p"
).dt.hour

day_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

df["Day_Number"] = df["Day of the week"].map(day_mapping)

# --------------------------------------------------
# 3. Select Features and Target
# --------------------------------------------------

X = df[
    [
        "Hour",
        "Day_Number",
        "CarCount",
        "BikeCount",
        "BusCount",
        "TruckCount"
    ]
]

y = df["Traffic Situation"]

# --------------------------------------------------
# 4. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 5. Create and Train Decision Tree
# --------------------------------------------------

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print("Decision Tree model trained successfully!")

# --------------------------------------------------
# 6. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------
# 7. Calculate Evaluation Metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

# --------------------------------------------------
# 8. Display Results
# --------------------------------------------------

print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# --------------------------------------------------
# 9. Classification Report
# --------------------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 10. Confusion Matrix
# --------------------------------------------------

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_test, y_pred)

print(cm)
# --------------------------------------------------
# 11. Visualization
# --------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

# --------------------------------------------------
# Graph 1: Traffic Situation Distribution
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    x="Traffic Situation",
    data=df
)

plt.title("Traffic Situation Distribution")
plt.xlabel("Traffic Situation")
plt.ylabel("Number of Samples")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# Graph 2: Confusion Matrix
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# Graph 3: Decision Tree
# --------------------------------------------------

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    max_depth=3
)

plt.title("Decision Tree for Traffic Situation Prediction")

plt.tight_layout()
plt.show()
