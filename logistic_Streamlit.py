import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Titanic Logistic Regression", layout="centered")
st.title("🚢 Titanic Survival Prediction (Logistic Regression)")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_titanic_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df

uploaded_file = st.file_uploader("Upload Titanic CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_titanic_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Check target column
# -----------------------------
target_col = "Survived"



# -----------------------------
# Select numeric feature columns
# -----------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

feature_cols = st.multiselect(
    "Select exactly 2 numeric feature columns",
    numeric_cols,
    default=["Age", "Fare"]
)

if len(feature_cols) == 2:
    # -----------------------------
    # Prepare data
    # -----------------------------
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # Train Logistic Regression
    # -----------------------------
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # -----------------------------
    # Accuracy
    # -----------------------------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Accuracy: {acc:.2f}")

    # -----------------------------
    # Decision boundary plot
    # -----------------------------
    st.subheader("📊 Decision Boundary")

    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.1),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.1)
    )

    plt.figure(figsize=(8, 6))
    plt.contourf(
        X1, X2,
        model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.3,
        cmap=plt.cm.Paired
    )

    plt.scatter(X_set[y_set == 0, 0], X_set[y_set == 0, 1], label="Did not survive", edgecolor='k')
    plt.scatter(X_set[y_set == 1, 0], X_set[y_set == 1, 1], label="Survived", edgecolor='k')

    plt.xlabel(f"{feature_cols[0]} (scaled)")
    plt.ylabel(f"{feature_cols[1]} (scaled)")
    plt.title("Titanic Logistic Regression Decision Boundary")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()
else:
    st.warning("⚠️ Please select exactly 2 feature columns")
