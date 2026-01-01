# --------------------------------------------------
# Fraud Detection Using Logistic Regression
# --------------------------------------------------
# This script downloads a chargeback/fraud dataset from KaggleHub,
# preprocesses it, trains a logistic regression model, evaluates it,
# and saves the predictions to a CSV file.
# It also includes some randomness in predictions based on probability.
# TARGET_COL: 'CBK' → 1 = fraud (Yes), 0 = not fraud (No)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import kagglehub
import os

# --------------------------------------------------
# 1. Load Dataset from KaggleHub
# --------------------------------------------------
# Download dataset using KaggleHub API and find CSV file
path = kagglehub.dataset_download("dmirandaalves/predict-chargeback-frauds-payment")

# Automatically find the CSV file in the downloaded folder
csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv_file))

print("Columns:", df.columns)
print("Number of rows:", len(df))
print(df.head())

# --------------------------------------------------
# 2. Define Columns
# --------------------------------------------------
TARGET_COL = "CBK"          # Column representing whether a transaction is a fraud (Yes/No)
ID_COL = "Unnamed: 0"       # Column representing transaction ID

# --------------------------------------------------
# 3. Data Cleaning & Feature Engineering
# --------------------------------------------------
# Convert target to lowercase and filter valid values
df[TARGET_COL] = df[TARGET_COL].str.lower()
df = df[df[TARGET_COL].isin(["yes", "no"])]

# Map target to binary numeric values (1 = fraud, 0 = not fraud)
df[TARGET_COL] = df[TARGET_COL].map({"yes": 1, "no": 0}).astype(np.float32)

# Convert Amount to numeric and fill missing values with median
df["Amount"] = pd.to_numeric(df["Amount"], errors='coerce')
df["Amount"] = df["Amount"].fillna(df["Amount"].median())

# Convert Date to datetime and fill missing dates with a default date
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df["Date"] = df["Date"].fillna(pd.to_datetime("2020-01-01"))

# Feature engineering: extract day, month, day_of_week from Date
df["day"] = df["Date"].dt.day
df["month"] = df["Date"].dt.month
df["day_of_week"] = df["Date"].dt.dayofweek

# Prepare features (X) and target (y)
X = df[["Amount", "day", "month", "day_of_week"]].astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32)
transaction_ids = df[ID_COL].values.astype(np.int32)

print("Prepared features:", X.shape, "Target:", y.shape)

# --------------------------------------------------
# 4. Train/Test Split
# --------------------------------------------------
# Split data into training and testing sets (80/20 split)
# stratify=y ensures same fraud ratio in train and test
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, transaction_ids,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 5. Feature Scaling
# --------------------------------------------------
# StandardScaler normalizes features to mean=0, std=1
# This helps logistic regression converge faster and improves performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# 6. Train Logistic Regression Model
# --------------------------------------------------
# class_weight='balanced' compensates for class imbalance (frauds are rare)
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# --------------------------------------------------
# 7. Predictions with Randomness
# --------------------------------------------------
# Predict fraud probabilities for the test set
y_prob = model.predict_proba(X_test)[:, 1]  # probability of fraud


# Predict based on threshold 0.5 instead of randomness
y_pred = (y_prob > 0.5).astype(int)


# --------------------------------------------------
# 8. Evaluation
# --------------------------------------------------
# Evaluate model using common classification metrics
print("\nModel Evaluation")
print("----------------")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")  # How many predicted frauds were correct
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")    # How many actual frauds were detected
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")        # Harmonic mean of precision & recall
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# --------------------------------------------------
# 9. Save Predictions to CSV
# --------------------------------------------------
# Save transaction ID,  predicted label 0(Not Fraud), 1(Fraud),  predicted fraud probability
output_df = pd.DataFrame({
    "transaction_id": id_test,
    "predicted_label": y_pred,
    "fraud_probability": y_prob
})

output_df.to_csv("fraud_predictions.csv", index=False)
print("\nSaved predictions to fraud_predictions.csv")
