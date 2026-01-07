import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# Page config
# =========================
st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

st.title("🚢 Titanic Survival Prediction (Logistic Regression)")

# =========================
# Upload CSV
# =========================
uploaded_file = st.file_uploader("Upload Titanic CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    st.write("📄 Columns in uploaded CSV:")
    st.write(list(df.columns))

    # Required columns
    required_cols = ['p_class', 'age', 'fare', 'survived']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
    else:
        # =========================
        # Prepare data
        # =========================
        X = df[['p_class', 'age', 'fare']].fillna(
            df[['p_class', 'age', 'fare']].mean()
        )
        y = df['survived']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # =========================
        # Train model
        # =========================
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Save model
        joblib.dump(model, "model.pkl")

        # Accuracy
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success("✅ Model trained and saved successfully!")
        st.write(f"🎯 Accuracy: **{acc:.2f}**")

        # =========================
        # Manual prediction
        # =========================
        st.subheader("🔮 Predict Survival")

        p_class = st.number_input("Passenger Class (1 / 2 / 3)", min_value=1, max_value=3, value=3)
        age = st.number_input("Age", min_value=0.0, value=25.0)
        fare = st.number_input("Fare", min_value=0.0, value=10.0)

        if st.button("Predict"):
            input_data = np.array([[p_class, age, fare]])
            result = model.predict(input_data)

            if result[0] == 1:
                st.success("🟢 Passenger Survived")
            else:
                st.error("🔴 Passenger Did NOT Survive")
