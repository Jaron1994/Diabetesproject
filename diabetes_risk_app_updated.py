
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
@st.cache_data
def load_data():
    file_path = "diabetes_data.csv"  # Ensure this file is in the same directory
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Prepare data for modeling
features = ["age", "hypertension", "heart_disease", "bmi"]
X = df[features]
y = df["diabetes"]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a simple logistic regression model
model = LogisticRegression()
model.fit(X_scaled, y)

# Streamlit UI
st.title("Diabetes Risk Predictor")
st.write("Enter your information below to see how your risk compares to the dataset.")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 30)
location = st.text_input("Location (State)", "Alabama")
race = st.selectbox("Race", ["African American", "Asian", "Caucasian", "Hispanic", "Other"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
smoking_history = st.selectbox("Smoking History", ["Never", "Former", "Current"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

# Convert categorical inputs
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0

# Prepare user input for prediction
user_data = np.array([[age, hypertension, heart_disease, bmi]])
user_data_scaled = scaler.transform(user_data)

# Predict probability
prob = model.predict_proba(user_data_scaled)[0][1]
average_prob = df["diabetes"].mean()

# Compute percentage difference from the dataset's average risk
risk_difference = ((prob - average_prob) / average_prob) * 100

# Display results
st.subheader("Prediction Results")
st.write(f"Your estimated probability of having diabetes: **{prob*100:.2f}%**")
st.write(f"Compared to the average population in the dataset, your risk is **{risk_difference:+.2f}%** {'higher' if risk_difference > 0 else 'lower'} than average.")

st.write("Disclaimer: This tool provides only an estimated risk based on available dataset trends. Consult a healthcare professional for accurate medical advice.")
