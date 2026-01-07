import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.title("🚢 Titanic Survival Prediction")

# 1️⃣ Upload CSV
uploaded_file = st.file_uploader("Upload Titanic CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ✅ Column safe fix: remove spaces, lowercase, replace spaces with underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Optional: show columns for debugging
    st.write("Columns in CSV after cleaning:", list(df.columns))

    # 2️⃣ Required columns
    required_cols = ['p_class','age','fare','survived']  # Corrected 'p_class'
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"CSV is missing columns: {missing_cols}")
    else:
        # 3️⃣ Prepare features safely
        X = df[['p_class','age','fare']].fillna(df[['p_class','age','fare']].mean())
        y = df['survived']

        # 4️⃣ Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 5️⃣ Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        st.success("✅ Model trained successfully!")

        # 6️⃣ Sample predictions
        predictions = model.predict(X_test)
        st.subheader("Sample Predictions")
        st.write(predictions[:10]) 
