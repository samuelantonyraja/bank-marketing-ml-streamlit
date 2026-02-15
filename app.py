import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

st.title("Bank Marketing Classification App")
st.subheader("Download Dataset")

col1, col2 = st.columns(2)

with col1:
    with open("bank-full.csv", "rb") as file:
        st.download_button(
            label="Download bank-full.csv",
            data=file,
            file_name="bank-full.csv",
            mime="text/csv"
        )

with col2:
    with open("bank.csv", "rb") as file:
        st.download_button(
            label="Download bank.csv",
            data=file,
            file_name="bank.csv",
            mime="text/csv"
        )

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

# Load selected model
model = joblib.load(f"model/{model_name}.pkl")

# Upload CSV
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';')

    df.columns = df.columns.str.strip()

    # Handle unknown values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace("unknown", df[col].mode()[0])

    # Convert target
    df["y"] = df["y"].map({"yes": 1, "no": 0})

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("y", axis=1)
    y = df["y"]

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y, predictions)
    st.text(report)

    # Confusion Matrix with numbers
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
