import pandas as pd

# Load dataset
df = pd.read_csv("cardio_train.csv", sep=";")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

import pandas as pd

df = pd.read_csv("cardio_train.csv", sep=";")

print("Missing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print("Number of duplicate rows:", df.duplicated().sum())

print("\nData Types")
print(df.dtypes)

print("\nTarget Column")
print(df["cardio"].value_counts())

import matplotlib.pyplot as plt

# Age distribution
plt.figure(figsize=(8, 5))
plt.hist(df["age"] / 365.25, bins=30)
plt.xlabel("Age")
plt.ylabel("Number of People")
plt.title("Age Distribution")
plt.show()

# Cardiovascular disease distribution
plt.figure(figsize=(6, 5))
df["cardio"].value_counts().plot(kind="bar")
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

X = df.drop("cardio", axis=1)
y = df["cardio"]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)
from sklearn.model_selection import train_test_split

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

from sklearn.preprocessing import StandardScaler

# Create scaler
scaler = StandardScaler()

# Fit scaler on training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Transform testing data
X_test_scaled = scaler.transform(X_test)

print("\nScaled Training Data Shape:")
print(X_train_scaled.shape)

print("\nScaled Testing Data Shape:")
print(X_test_scaled.shape)

from sklearn.linear_model import LogisticRegression

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_scaled, y_train)

print("\nModel Training Completed")