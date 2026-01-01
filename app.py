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

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------------------------------
    # Dataset Preview
    # -------------------------------
    st.subheader("📄 Dataset Preview")
    rows_to_show = st.slider("Number of rows to preview", min_value=5, max_value=50, value=20)
    st.dataframe(df.head(rows_to_show))

    # -------------------------------
    # Column Definitions
    # -------------------------------
    TARGET_COL = "CBK"
    ID_COL = "Sr No"

    try:
        # -------------------------------
        # Data Cleaning & Feature Engineering
        # -------------------------------
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

        # -------------------------------
        # Feature Selection
        # -------------------------------
        st.subheader("⚙️ Select Features for Model")
        feature_options = ["Amount", "day", "month", "day_of_week"]
        selected_features = st.multiselect(
            "Select features", feature_options, default=feature_options
        )

        X = df[selected_features].astype(np.float32)
        y = df[TARGET_COL].values.astype(np.float32)
        transaction_ids = df[ID_COL].values.astype(np.int32)

        # -------------------------------
        # Train/Test Split
        # -------------------------------
        test_size = st.slider("Test Set Size (%)", 10, 50, 20)
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X, y, transaction_ids,
            test_size=test_size/100,
            random_state=42,
            stratify=y
        )

        # -------------------------------
        # Scaling
        # -------------------------------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # -------------------------------
        # Train Model
        # -------------------------------
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(X_train, y_train)

        # -------------------------------
        # Probability Threshold
        # -------------------------------
        threshold = st.slider("Fraud Probability Threshold", 0.1, 0.9, 0.5)

        # -------------------------------
        # Predictions
        # -------------------------------
        X_scaled_full = scaler.transform(X)
        y_prob_full = model.predict_proba(X_scaled_full)[:, 1]
        y_pred_full = (y_prob_full > threshold).astype(int)

        output_df = pd.DataFrame({
            "transaction_id": transaction_ids,
            "predicted_label": y_pred_full,
            "fraud_probability": y_prob_full
        })

        # Add risk level for table display
        output_df["risk_level"] = np.where(output_df["fraud_probability"] > 0.7, "High ⚠️", "Low ✅")

        # -------------------------------
        # Function to highlight high-risk transactions
        # -------------------------------
        def highlight_high_risk(val):
            color = 'background-color: #ffcccc' if "High" in str(val) else ''
            return color

        st.subheader("🔍 Prediction Preview (Interactive Table)")
        st.dataframe(
            output_df.style.applymap(highlight_high_risk, subset=["risk_level"]),
            use_container_width=True
        )

        # -------------------------------
        # Download Predictions
        # -------------------------------
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

