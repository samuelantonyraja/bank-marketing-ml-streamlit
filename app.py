import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

st.title("Bank Marketing Classification App")

# Load scaler
scaler = joblib.load("model/scaler.pkl")

# Model selection
model_name = st.selectbox(
    "Select Model",
    [
        "Logistic_Regression",
        "Decision_Tree",
        "KNN",
        "Naive_Bayes",
        "Random_Forest",
        "XGBoost"
    ]
)

# Load model
model = joblib.load(f"model/{model_name}.pkl")

# Upload CSV
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';')

    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace("unknown", df[col].mode()[0])

    df["y"] = df["y"].map({"yes":1, "no":0})

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("y", axis=1)
    y = df["y"]

    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)

    st.subheader("Classification Report")
    st.text(classification_report(y, predictions))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots()
    ax.matshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
