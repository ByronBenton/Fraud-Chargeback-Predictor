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

    # --------------------------------------------------
    # Dataset Preview
    # --------------------------------------------------
    st.subheader("📄 Dataset Preview")
    rows_to_show = st.slider("Rows to preview", 5, 50, 20)
    st.dataframe(df.head(rows_to_show), use_container_width=True)

    TARGET_COL = "CBK"
    ID_COL = "Sr No"

    try:
        # --------------------------------------------------
        # Data Cleaning & Feature Engineering
        # --------------------------------------------------
        df[TARGET_COL] = df[TARGET_COL].str.lower()
        df = df[df[TARGET_COL].isin(["yes", "no"])]
        df[TARGET_COL] = df[TARGET_COL].map({"yes": 1, "no": 0})

        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
        df["Amount"].fillna(df["Amount"].median(), inplace=True)

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Date"].fillna(pd.to_datetime("2020-01-01"), inplace=True)

        df["day"] = df["Date"].dt.day
        df["month"] = df["Date"].dt.month
        df["day_of_week"] = df["Date"].dt.dayofweek

        X = df[["Amount", "day", "month", "day_of_week"]]
        y = df[TARGET_COL]
        transaction_ids = df[ID_COL]

        # --------------------------------------------------
        # Train/Test Split
        # --------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
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
        # Metrics
        # --------------------------------------------------
        y_prob_test = model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_prob_test > 0.5).astype(int)

        st.subheader("📊 Model Performance")
        st.write(f"Accuracy: **{accuracy_score(y_test, y_pred_test):.4f}**")
        st.write(f"Precision: **{precision_score(y_test, y_pred_test, zero_division=0):.4f}**")
        st.write(f"Recall: **{recall_score(y_test, y_pred_test, zero_division=0):.4f}**")
        st.write(f"F1-score: **{f1_score(y_test, y_pred_test, zero_division=0):.4f}**")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred_test))

        # --------------------------------------------------
        # Predict on FULL Dataset
        # --------------------------------------------------
        X_scaled_full = scaler.transform(X)
        y_prob_full = model.predict_proba(X_scaled_full)[:, 1]
        y_pred_full = (y_prob_full > 0.5).astype(int)

        output_df = pd.DataFrame({
            "transaction_id": transaction_ids,
            "prediction": np.where(y_pred_full == 1, "Fraud", "Not Fraud"),
            "fraud_probability": y_prob_full
        })

        output_df["risk"] = np.where(
            output_df["fraud_probability"] > 0.7, "High ⚠️", "Low ✅"
        )

        # Add selection column
        output_df["Select"] = False

        # Sort for demo polish
        #output_df = output_df.sort_values("fraud_probability", ascending=False)

        # --------------------------------------------------
        # INTERACTIVE TABLE
        # --------------------------------------------------
        st.subheader("🔍 Prediction Table (Click a row)")

        edited_df = st.data_editor(
            output_df,
            use_container_width=True,
            disabled=["transaction_id", "prediction", "fraud_probability", "risk"],
        )

        # --------------------------------------------------
        # Show details for selected row
        # --------------------------------------------------
        selected = edited_df[edited_df["Select"] == True]

        if not selected.empty:
            row = selected.iloc[0]
            st.success(
                f"""
                **Transaction ID:** {row['transaction_id']}  
                **Prediction:** {row['prediction']}  
                **Fraud Probability:** {row['fraud_probability']:.2f}  
                **Risk Level:** {row['risk']}
                """
            )

        # --------------------------------------------------
        # Download
        # --------------------------------------------------
        csv = output_df.drop(columns=["Select"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download fraud_predictions.csv",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error("❌ Error processing file.")
        st.write(e)
