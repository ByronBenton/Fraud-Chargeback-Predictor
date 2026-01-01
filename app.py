import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Fraud Detection App", layout="centered")

st.title("💳 Fraud Detection using Logistic Regression")
st.write("Upload a CSV file to detect fraudulent transactions.")

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Column Definitions
    # --------------------------------------------------
    TARGET_COL = "CBK"
    ID_COL = "Sr No"   # ✅ UPDATED COLUMN NAME

    try:
        # --------------------------------------------------
        # Data Cleaning & Feature Engineering
        # --------------------------------------------------
        df[TARGET_COL] = df[TARGET_COL].str.lower()
        df = df[df[TARGET_COL].isin(["yes", "no"])]
        df[TARGET_COL] = df[TARGET_COL].map({"yes": 1, "no": 0}).astype(np.float32)

        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
        df["Amount"].fillna(df["Amount"].median(), inplace=True)

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Date"].fillna(pd.to_datetime("2020-01-01"), inplace=True)

        df["day"] = df["Date"].dt.day
        df["month"] = df["Date"].dt.month
        df["day_of_week"] = df["Date"].dt.dayofweek

        X = df[["Amount", "day", "month", "day_of_week"]].astype(np.float32)
        y = df[TARGET_COL].values.astype(np.float32)
        transaction_ids = df[ID_COL].values.astype(np.int32)

        # --------------------------------------------------
        # Train/Test Split
        # --------------------------------------------------
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X, y, transaction_ids,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # --------------------------------------------------
        # Scaling
        # --------------------------------------------------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # --------------------------------------------------
        # Train Model
        # --------------------------------------------------
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(X_train, y_train)

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------
        st.subheader("📊 Model Performance")
        st.write(f"Accuracy: **{accuracy_score(y_test, y_pred):.4f}**")
        st.write(f"Precision: **{precision_score(y_test, y_pred):.4f}**")
        st.write(f"Recall: **{recall_score(y_test, y_pred):.4f}**")
        st.write(f"F1-score: **{f1_score(y_test, y_pred):.4f}**")

        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))

        # --------------------------------------------------
        # Output CSV
        # --------------------------------------------------
        output_df = pd.DataFrame({
            "transaction_id": id_test,
            "predicted_label": y_pred,
            "fraud_probability": y_prob
        })

        csv = output_df.to_csv(index=False).encode("utf-8")

        st.subheader("⬇️ Download Predictions, 0-Not Fraud, 1-Fraud")
        st.download_button(
            label="Download fraud_predictions.csv",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("❌ Error processing file. Please check CSV format.")
        st.write(e)
