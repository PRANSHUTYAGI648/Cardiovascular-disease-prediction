import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Load dataset

df = pd.read_csv("cardio_train.csv", sep=";")


# Display first 5 rows

print("First 5 Rows:")
print(df.head())


# Dataset shape

print("\nDataset Shape:")
print(df.shape)


# Dataset information

print("\nDataset Information:")
df.info()


# Statistical summary

print("\nStatistical Summary:")
print(df.describe())


# Missing values

print("\nMissing Values:")
print(df.isnull().sum())


# Duplicate rows

print("\nDuplicate Rows:")
print("Number of duplicate rows:", df.duplicated().sum())


# Data types

print("\nData Types:")
print(df.dtypes)


# Target column

print("\nTarget Column:")
print(df["cardio"].value_counts())


# Age distribution

plt.figure(figsize=(8, 5))
plt.hist(df["age"] / 365.25, bins=30)

plt.xlabel("Age")
plt.ylabel("Number of People")
plt.title("Age Distribution")

plt.show()


# Cardiovascular disease distribution

plt.figure(figsize=(6, 5))
df["cardio"].value_counts().sort_index().plot(kind="bar")

plt.xlabel("Cardiovascular Disease")
plt.ylabel("Number of People")
plt.title("Cardiovascular Disease Distribution")
plt.xticks(rotation=0)

plt.show()


# Cholesterol distribution

plt.figure(figsize=(6, 5))
df["cholesterol"].value_counts().sort_index().plot(kind="bar")

plt.xlabel("Cholesterol Level")
plt.ylabel("Number of People")
plt.title("Cholesterol Distribution")
plt.xticks(rotation=0)

plt.show()


# Separate features and target

X = df.drop(["id", "cardio"], axis=1)
y = df["cardio"]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


# Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# Feature scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled Training Data Shape:")
print(X_train_scaled.shape)

print("\nScaled Testing Data Shape:")
print(X_test_scaled.shape)


# Create Logistic Regression model

model = LogisticRegression(max_iter=1000)

# Train the model

model.fit(X_train_scaled, y_train)

print("\nModel Training Completed")


# Make predictions

y_pred = model.predict(X_test_scaled)

print("\nPrediction Completed")
print("First 20 Predictions:")
print(y_pred[:20])

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nModel Accuracy Percentage:")
print(accuracy * 100, "%")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.xticks([0, 1], ["No Disease", "Disease"])
plt.yticks([0, 1], ["No Disease", "Disease"])

plt.colorbar()

plt.show()

from sklearn.metrics import classification_report

report = classification_report(
    y_test,
    y_pred,
    target_names=["No Disease", "Disease"]
)

print("\nClassification Report:")
print(report)

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Random Forest

rf_model.fit(X_train, y_train)

print("\nRandom Forest Training Completed")

# Make predictions

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Prediction Completed")

print("First 20 Random Forest Predictions:")
print(rf_pred[:20])

from sklearn.metrics import accuracy_score

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)

print("\nRandom Forest Accuracy Percentage:")
print(rf_accuracy * 100, "%")
print("\nModel Comparison")
print("----------------")

print("Logistic Regression Accuracy:", accuracy * 100, "%")
print("Random Forest Accuracy:", rf_accuracy * 100, "%")

if rf_accuracy > accuracy:
    print("\nBest Model: Random Forest")
elif accuracy > rf_accuracy:
    print("\nBest Model: Logistic Regression")
else:
    print("\nBoth Models Have Same Accuracy")


with open("cardio_random_forest_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

print("\nBest Model Saved Successfully")
print("Model File: cardio_random_forest_model.pkl")

import numpy as np

import numpy as np

new_patient = np.array([[
    50 * 365.25,
    2,
    170,
    70,
    120,
    80,
    1,
    1,
    0,
    0,
    1
]])

new_prediction = rf_model.predict(new_patient)

print("\nNew Patient Prediction:")

if new_prediction[0] == 1:
    print("Prediction: Cardiovascular Disease")
else:
    print("Prediction: No Cardiovascular Disease")

new_prediction = rf_model.predict(new_patient)

print("\nNew Patient Prediction:")

if new_prediction[0] == 1:
    print("Prediction: Cardiovascular Disease")
else:
    print("Prediction: No Cardiovascular Disease")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

rf_cm = confusion_matrix(y_test, rf_pred)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)

plt.figure(figsize=(6, 5))
plt.imshow(rf_cm)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")

plt.xticks([0, 1], ["No Disease", "Disease"])
plt.yticks([0, 1], ["No Disease", "Disease"])

plt.colorbar()

plt.show()  

from sklearn.metrics import classification_report

rf_report = classification_report(
    y_test,
    rf_pred,
    target_names=["No Disease", "Disease"]
)

print("\nRandom Forest Classification Report:")
print(rf_report)

import pandas as pd
import matplotlib.pyplot as plt

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)

feature_importance = feature_importance.sort_values(ascending=False)

print("\nRandom Forest Feature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind="bar")

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()