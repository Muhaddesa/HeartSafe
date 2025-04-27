# 💓 Heart Disease Prediction App

## Overview
This project is a **machine learning-based web application** developed with **Streamlit** that predicts the likelihood of heart disease based on patient information.  
It provides a simple and interactive interface for users to input medical parameters and receive predictions in real-time.

## Features
- 🚀 Predicts presence of heart disease with a trained machine learning model
- 🎛️ Scales numerical features and encodes categorical features manually inside the app
- 🎨 Custom Streamlit UI styling for better user experience
- 📈 Displays prediction probabilities clearly

## Technologies Used
- Python
- Streamlit
- scikit-learn
- pandas
- numpy

## Model
- Trained different models such as Random Forest Classifier, KNN, SVM while all had significant performance, I wanted a more robust model;
  so I used a Voting Classifier type of soft voting.
- Input features include:  
  `age`, `sex`, `chest pain type`, `resting blood pressure`, `serum cholesterol`, `fasting blood sugar`, `resting ECG`, `max heart rate`, `exercise-induced angina`, `ST depression`, `slope`, `major vessels`, and `thalassemia`

## How to Run Locally
## How to Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/Muhaddesa/HeartSafe.git
   cd HeartSafe
   Install the required Python packages:

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit App:
   ```bash
   streamlit run streamlit-app.py

4. Open the link provided in the terminal (usually http://localhost:8501) to use the app in your browser.
