import joblib
import numpy as np
import streamlit as st

# Load Models
reg_model = joblib.load("reg_model.pkl")
clf_model = joblib.load("clf_model.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎓 STEM Admission & Student Analytics System")
st.write(
    "Predict readiness score, admission status, and student persona using Machine Learning."
)

# User Inputs
math = st.slider("Math Score", 0, 100, 85)
science = st.slider("Science Score", 0, 100, 80)
english = st.slider("English Score", 0, 100, 75)

if st.button("Analyze Student Profile"):
    input_data = np.array([[math, science, english]])

    # 1. Regression Prediction
    predicted_score = reg_model.predict(input_data)[0]

    # 2. Classification Prediction
    admission_status = clf_model.predict(input_data)[0]

    # 3. Student Profiling
    avg_score = (math + science + english) / 3

    if avg_score >= 82:
        persona = "High Achiever 🌟 (Top Performance)"
    elif (math + science) / 2 >= 80:
        persona = "STEM Focused 🔬 (Strong in Math & Science)"
    else:
        persona = "Needs Academic Support 📈 (Requires Skill Building)"

    st.subheader("📊 Analysis Results:")
    st.write(f"**Predicted Readiness Score (Regression):** {predicted_score:.2f}%")

    if admission_status == 1:
        st.success("**Admission Status (Classification):** Accepted 🎉")
    else:
        st.error("**Admission Status (Classification):** Rejected ❌")

    st.info(f"**Student Persona (Clustering/Profiling):** {persona}")
