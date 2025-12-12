# app.py - Student Score Predictor with plot
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Score Predictor", page_icon="🎯", layout="centered")

st.title("Student Score Predictor")
st.write("Enter how many hours a student studies and the app will predict the expected score (based on a simple linear model).")

@st.cache_resource
def load_model_and_data():
    model = joblib.load("linear_regressor_joblib.pkl")
    df = pd.read_csv("student_scores.csv")
    return model, df

model, df = load_model_and_data()

# --- Left: Input / Right: Plot
col1, col2 = st.columns([1,1])

with col1:
    hours = st.number_input("Hours studied", min_value=0.0, max_value=24.0, value=5.0, step=0.25, format="%.2f")
    if st.button("Predict"):
        X_new = pd.DataFrame([[hours]], columns=["Hours"])
        pred = model.predict(X_new)[0]
        st.success(f"Predicted Score: {pred:.2f} / 100")
        st.write("This is a simple linear model trained on a tiny sample dataset for learning purposes.")

    st.markdown("---")
    st.write("Quick examples:")
    if st.button("Show 3 hrs"):
        st.info(f"Predicted: {model.predict(pd.DataFrame([[3.0]], columns=['Hours']))[0]:.2f}")
    if st.button("Show 6 hrs"):
        st.info(f"Predicted: {model.predict(pd.DataFrame([[6.0]], columns=['Hours']))[0]:.2f}")
    if st.button("Show 9 hrs"):
        st.info(f"Predicted: {model.predict(pd.DataFrame([[9.0]], columns=['Hours']))[0]:.2f}")

with col2:
    st.write("Scatter plot with regression line")
    # Prepare line
    X = df[["Hours"]]
    y = df["Score"]
    X_line = np.linspace(X["Hours"].min(), X["Hours"].max(), 200).reshape(-1,1)
    # ensure model works if it's a pipeline/needs transform — we trained a plain LinearRegression on DataFrame
    # use DataFrame with same column name for prediction
    X_line_df = pd.DataFrame(X_line, columns=["Hours"])
    y_line = model.predict(X_line_df)

    fig, ax = plt.subplots(figsize=(5,3.5))
    ax.scatter(X["Hours"], y, label="Data")
    ax.plot(X_line, y_line, linewidth=2, label="Regression line")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Score")
    ax.set_title("Hours vs Score")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

st.markdown("---")
st.write("Model file: `linear_regressor_joblib.pkl`  •  Data: `student_scores.csv`")
