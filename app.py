import streamlit as st
import numpy as np
import joblib 

# Load models and scaler
clf1 = joblib.load("logistic_model.pkl")
clf2 = joblib.load("svm_model.pkl")
clf3 = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
weights = np.load("model_weights.npy")

# Weighted Median logic
def weighted_median_prediction(probs, weights):
    sorted_indices = np.argsort(probs)
    sorted_preds = probs[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cum_weights = np.cumsum(sorted_weights)
    median_idx = np.where(cum_weights >= 0.5)[0][0]
    return 1 if sorted_preds[median_idx] >= 0.5 else 0

# Streamlit UI
st.title("🧠 WMC Doctor – Diabetes Prediction")

st.markdown("Enter medical info below to check your risk:")

pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin Level", 0, 1000)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    # Step 1: Preprocess
    user_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    user_scaled = scaler.transform(user_input)

    # Step 2: Get predictions from base models
    p1 = clf1.predict_proba(user_scaled)[0][1]
    p2 = clf2.predict_proba(user_scaled)[0][1]
    p3 = clf3.predict_proba(user_scaled)[0][1]

    probs = np.array([p1, p2, p3])
    prediction = weighted_median_prediction(probs, weights)

    # Step 3: Show result
    if prediction == 1:
        st.error("⚠️ Likely Diabetic. Please consult a doctor.")
    else:
        st.success("✅ Healthy. Keep up the good work!")
