import streamlit as st
import joblib
import numpy as np

# Load model and scaler
voting_model = joblib.load('voting_classifier.pkl')
scaler = joblib.load('scaler.pkl')

# App title and header
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.markdown("<h1 style='text-align: center; color: crimson;'>💓 Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center;'>
This intelligent application uses a <b>Voting Classifier</b> model to predict the likelihood of heart disease.  
Enter patient details below and click <b>Predict</b> to view the results.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Layout: 3 columns for input
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("🧓 Age", 1, 120, 45)
    trestbps = st.slider("💉 Resting Blood Pressure (mm Hg)", 50, 300, 120)
    chol = st.slider("🥩 Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.radio("🩸 Fasting Blood Sugar > 120 mg/dl?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)

with col2:
    sex = st.radio("⚧️ Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", horizontal=True)
    cp = st.selectbox("💓 Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"][x])
    thalach = st.number_input(
        "❤️ Enter your Maximum Heart Rate Achieved (thalach)",
        min_value=50,
        max_value=250,
        value=94,
        step=1,
        help="This is the highest heart rate you achieved during a stress test."
    )
    exang = st.radio("🏃 Exercise-Induced Angina?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)

with col3:
    restecg = st.selectbox("📈 Resting ECG Results", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"][x])
    oldpeak = st.number_input(
        "📉 Enter ST Depression (oldpeak):",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.1,
        format="%.1f",
        help="ST depression induced by exercise relative to rest. Higher values may indicate higher risk."
    )
    slope = st.selectbox("⛰️ Slope of ST Segment", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.selectbox("🩺 # of Major Vessels Colored", [0, 1, 2, 3, 4])
    thal = st.selectbox("🧬 Thalassemia", [0, 1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])

st.markdown("---")

# Prediction
if st.button("🔍 Predict", use_container_width=True):
    input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_features)

    prediction = voting_model.predict(input_scaled)
    prob = voting_model.predict_proba(input_scaled)[0][1]  # Probability of Heart Disease

    st.markdown("## 🧾 Results", unsafe_allow_html=True)

    # Centered Result Text
    if prediction[0] == 1:
        result_text = "<h2 style='text-align: center; color: red;'>⚠️ Heart Disease Detected</h2>"
    else:
        result_text = "<h2 style='text-align: center; color: green;'>✅ No Heart Disease Detected</h2>"

    st.markdown(result_text, unsafe_allow_html=True)

    if prediction[0] == 0:
        st.balloons()  # 🎈 Animation when no disease is detected

    # Doctor Advice (Always shown)
    st.markdown("### 🩺 Doctor's Preventive Care Advice", unsafe_allow_html=True)

    if prediction[0] == 1:
        if prob < 0.4:
            st.markdown("""
            - 🥗 Focus on a healthy diet rich in vegetables and fruits.
            """)
        elif prob < 0.7:
            st.markdown("""
            - 🚶 Exercise regularly (at least 150 min per week).
            - 🩺 Monitor blood pressure and cholesterol levels.
            """)
        else:
            st.markdown("""
            - 🚨 Immediate doctor consultation recommended.
            - 💊 Strictly follow prescribed medications.
            - 🛌 Reduce workload and prioritize heart health.
            """)
    else:
        st.markdown("""
        - 🥗 Maintain your current healthy lifestyle!
        - 🚭 Avoid smoking and limit processed foods.
        """)

# Feature explanation
with st.expander("🧠 Learn More About Each Feature"):
    st.markdown("""
    - **Age**: Age in years  
    - **Sex**: Female (0), Male (1)  
    - **Chest Pain Type**:  
        - 0: Typical Angina  
        - 1: Atypical Angina  
        - 2: Non-Anginal Pain  
        - 3: Asymptomatic  
    - **Resting Blood Pressure**: mm Hg  
    - **Cholesterol**: mg/dl  
    - **Fasting Blood Sugar**: > 120 mg/dl (1 = Yes)  
    - **Resting ECG**:  
        - 0: Normal  
        - 1: ST-T wave abnormality  
        - 2: Left Ventricular Hypertrophy  
    - **Max Heart Rate Achieved**  
    - **Exercise-Induced Angina**  
    - **Oldpeak**: ST depression induced by exercise  
    - **Slope**:  
        - 0: Upsloping  
        - 1: Flat  
        - 2: Downsloping  
    - **CA**: Major vessels (0-4)  
    - **Thal**:  
        - 0: Normal  
        - 1: Fixed Defect  
        - 2: Reversible Defect  
        - 3: Unknown
    """)
