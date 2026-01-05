import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

st.set_page_config(page_title="Logistic Regression App", layout="wide")
st.title("📈 Logistic Regression Streamlit App")

# ----------------------------
# Default demo dataset (Social Network Ads)
# ----------------------------
def load_demo_data():
    data = {
        "Age": [19, 35, 26, 27, 19, 27, 27, 32, 25, 35],
        "EstimatedSalary": [19000, 20000, 43000, 57000, 76000, 58000, 84000, 150000, 33000, 65000],
        "Purchased": [0, 0, 0, 0, 0, 1, 1, 1, 0, 1]
    }
    return pd.DataFrame(data)

# ----------------------------
# Step 1: Upload CSV
# ----------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV Loaded Successfully!")
else:
    st.info("No CSV uploaded. Using demo dataset.")
    df = load_demo_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ----------------------------
# Step 2: Select Features & Target
# ----------------------------
all_columns = df.columns.tolist()
target_col = st.selectbox("Select Target Column", all_columns, index=len(all_columns)-1)
feature_cols = st.multiselect("Select Feature Columns (2 only for plot)", all_columns, default=all_columns[:2])

if len(feature_cols) != 2:
    st.warning("Please select exactly 2 features for plotting!")
else:
    X = df[feature_cols]
    y = df[target_col]

    # ----------------------------
    # Step 3: Train-Test Split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # ----------------------------
    # Step 4: Feature Scaling
    # ----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------
    # Step 5: Train Logistic Regression
    # ----------------------------
    log_model = LogisticRegression()
    log_model.fit(X_train_scaled, y_train)

    # ----------------------------
    # Step 6: Model Evaluation
    # ----------------------------
    y_pred = log_model.predict(X_test_scaled)
    st.subheader("Model Performance")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    st.write("**Confusion Matrix:**\n", confusion_matrix(y_test, y_pred))
    st.write("**Classification Report:**\n", classification_report(y_test, y_pred))

    # ----------------------------
    # Step 7: Save Model
    # ----------------------------
    joblib.dump(log_model, "logistic_regression_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    st.success("✅ Model and Scaler saved using joblib!")

    # ----------------------------
    # Step 8: Make Predictions
    # ----------------------------
    st.subheader("Make New Prediction")
    new_feature1 = st.number_input(f"Enter {feature_cols[0]}", float(X[feature_cols[0]].min()), float(X[feature_cols[0]].max()))
    new_feature2 = st.number_input(f"Enter {feature_cols[1]}", float(X[feature_cols[1]].min()), float(X[feature_cols[1]].max()))
    if st.button("Predict"):
        new_data = np.array([[new_feature1, new_feature2]])
        new_data_scaled = scaler.transform(new_data)
        prediction = log_model.predict(new_data_scaled)
        st.success(f"Prediction: {prediction[0]}")

    # ----------------------------
    # Step 9: Decision Boundary Plot
    # ----------------------------
    st.subheader("Decision Boundary Plot")
    X_set = X_test_scaled
    y_set = y_test.values

    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01)
    )

    plt.figure(figsize=(10,6))
    plt.contourf(
        X1, X2,
        log_model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.3,
        cmap=ListedColormap(('red','green'))
    )
    plt.scatter(X_set[y_set==0,0], X_set[y_set==0,1], c='red', label='Not Purchased')
    plt.scatter(X_set[y_set==1,0], X_set[y_set==1,1], c='green', label='Purchased')
    plt.xlabel(f"{feature_cols[0]} (scaled)")
    plt.ylabel(f"{feature_cols[1]} (scaled)")
    plt.title("Logistic Regression Decision Boundary")
    plt.legend()
    plt.grid(False)

    st.pyplot(plt.gcf())
