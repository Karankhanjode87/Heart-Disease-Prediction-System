import joblib
import pandas as pd


# Load trained model
model = joblib.load("ml_model/heart_disease_model.pkl")

print("Model loaded successfully!")


# Sample patient
patient = pd.DataFrame([{
    "Age": 60,
    "Gender": "Male",
    "Weight": 80,
    "Height": 170,
    "BMI": 27.7,
    "Smoking": "No",
    "Alcohol_Intake": "No",
    "Physical_Activity": "Moderate",
    "Diet": "Healthy",
    "Stress_Level": "Medium",
    "Hypertension": 1,
    "Diabetes": 1,
    "Hyperlipidemia": 1,
    "Family_History": 1,
    "Previous_Heart_Attack": 0,
    "Systolic_BP": 150,
    "Diastolic_BP": 95,
    "Heart_Rate": 85,
    "Blood_Sugar_Fasting": 140,
    "Cholesterol_Total": 250
}])


# Make prediction
prediction = model.predict(patient)[0]

# Get probability
probability = model.predict_proba(patient)[0][1]


print("\nPrediction:", prediction)
print("Heart Disease Probability:", round(probability * 100, 2), "%")