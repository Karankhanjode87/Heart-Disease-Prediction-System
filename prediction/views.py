from django.shortcuts import render, redirect, get_object_or_404
import joblib
import pandas as pd
from pathlib import Path

from .models import Feedback, Prediction

from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# LOAD TRAINED ML MODEL
# ============================================================

model = joblib.load(
    BASE_DIR / "ml_model" / "heart_disease_model.pkl"
)


# ============================================================
# PATIENT PREDICTION
# ============================================================

@login_required(login_url="login")
def home(request):

    if request.method == "POST":

        # ----------------------------------------------------
        # Get patient name
        # ----------------------------------------------------

        patient_name = request.POST.get("Patient_Name", "").strip()

        if not patient_name:
            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Patient name is required."
                }
            )

        if len(patient_name) > 100:
            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Patient name must be 100 characters or less."
                }
            )

        # ----------------------------------------------------
        # Get form values safely
        # ----------------------------------------------------

        try:

            age = int(request.POST.get("Age"))
            weight = float(request.POST.get("Weight"))
            height = float(request.POST.get("Height"))
            bmi = float(request.POST.get("BMI"))

            hypertension = int(
                request.POST.get("Hypertension")
            )

            diabetes = int(
                request.POST.get("Diabetes")
            )

            hyperlipidemia = int(
                request.POST.get("Hyperlipidemia")
            )

            family_history = int(
                request.POST.get("Family_History")
            )

            previous_heart_attack = int(
                request.POST.get("Previous_Heart_Attack")
            )

            systolic_bp = int(
                request.POST.get("Systolic_BP")
            )

            diastolic_bp = int(
                request.POST.get("Diastolic_BP")
            )

            heart_rate = int(
                request.POST.get("Heart_Rate")
            )

            blood_sugar = float(
                request.POST.get("Blood_Sugar_Fasting")
            )

            cholesterol = float(
                request.POST.get("Cholesterol_Total")
            )

        except (TypeError, ValueError):

            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Please enter valid numeric values in all fields."
                }
            )

        # ====================================================
        # NUMERIC VALIDATION
        # ====================================================

        if not 1 <= age <= 120:
            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Age must be between 1 and 120."
                }
            )

        if not 1 <= weight <= 300:
            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Weight must be between 1 and 300 kg."
                }
            )

        if not 30 <= height <= 250:
            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Height must be between 30 and 250 cm."
                }
            )

        if not 5 <= bmi <= 80:
            return render(
                request,
                "prediction/home.html",
                {
                    "error": "BMI must be between 5 and 80."
                }
            )

        if not 50 <= systolic_bp <= 300:
            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Systolic BP must be between 50 and 300."
                }
            )

        if not 30 <= diastolic_bp <= 200:
            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Diastolic BP must be between 30 and 200."
                }
            )

        if not 30 <= heart_rate <= 250:
            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Heart rate must be between 30 and 250."
                }
            )

        if not 1 <= blood_sugar <= 1000:
            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Blood sugar value is outside the allowed range."
                }
            )

        if not 50 <= cholesterol <= 1000:
            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Cholesterol value is outside the allowed range."
                }
            )

        # ----------------------------------------------------
        # Validate binary fields
        # ----------------------------------------------------

        binary_values = [
            hypertension,
            diabetes,
            hyperlipidemia,
            family_history,
            previous_heart_attack
        ]

        if any(value not in [0, 1] for value in binary_values):

            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Invalid yes/no value received."
                }
            )

        # ====================================================
        # VALIDATE DROPDOWN VALUES
        # ====================================================

        gender = request.POST.get("Gender")

        smoking = request.POST.get("Smoking")

        alcohol_intake = request.POST.get(
            "Alcohol_Intake"
        )

        physical_activity = request.POST.get(
            "Physical_Activity"
        )

        diet = request.POST.get("Diet")

        stress_level = request.POST.get(
            "Stress_Level"
        )

        # ----------------------------------------------------
        # Allowed values
        # ----------------------------------------------------

        if gender not in ["Male", "Female"]:

            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Invalid gender value."
                }
            )

        if smoking not in ["Yes", "No"]:

            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Invalid smoking value."
                }
            )

        if alcohol_intake not in [
            "None",
            "Low",
            "Moderate",
            "High",
            "Unknown"
        ]:

            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Invalid alcohol intake value."
                }
            )

        if physical_activity not in [
            "Low",
            "Moderate",
            "High"
        ]:

            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Invalid physical activity value."
                }
            )

        if diet not in [
            "Poor",
            "Average",
            "Good"
        ]:

            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Invalid diet value."
                }
            )

        if stress_level not in [
            "Low",
            "Moderate",
            "High"
        ]:

            return render(
                request,
                "prediction/home.html",
                {
                    "error": "Invalid stress level value."
                }
            )

        # ====================================================
        # PREPARE DATA FOR ML MODEL
        # ====================================================

        data = {

            "Age": age,

            "Gender": gender,

            "Weight": weight,

            "Height": height,

            "BMI": bmi,

            "Smoking": smoking,

            "Alcohol_Intake": alcohol_intake,

            "Physical_Activity": physical_activity,

            "Diet": diet,

            "Stress_Level": stress_level,

            "Hypertension": hypertension,

            "Diabetes": diabetes,

            "Hyperlipidemia": hyperlipidemia,

            "Family_History": family_history,

            "Previous_Heart_Attack": previous_heart_attack,

            "Systolic_BP": systolic_bp,

            "Diastolic_BP": diastolic_bp,

            "Heart_Rate": heart_rate,

            "Blood_Sugar_Fasting": blood_sugar,

            "Cholesterol_Total": cholesterol,
        }

        print("Patient data received:")
        print(data)

        # ====================================================
        # CONVERT DATA INTO DATAFRAME
        # ====================================================

        patient_df = pd.DataFrame([data])

        # ====================================================
        # MAKE PREDICTION
        # ====================================================

        prediction = model.predict(patient_df)[0]

        # ====================================================
        # CALCULATE PROBABILITY
        # ====================================================

        probability = (
            model.predict_proba(patient_df)[0][1] * 100
        )

        # ====================================================
        # SAVE PREDICTION
        # ====================================================

        Prediction.objects.create(

            user=request.user,

            patient_name=patient_name,

            age=age,

            gender=gender,

            weight=weight,

            height=height,

            bmi=bmi,

            prediction=int(prediction),

            probability=float(probability)
        )

        print("Prediction:", prediction)

        print(
            "Heart Disease Probability:",
            probability
        )

        # ====================================================
        # SEND RESULT PAGE
        # ====================================================

        context = {

            "patient_name": patient_name,

            "prediction": prediction,

            "probability": probability
        }

        return render(
            request,
            "prediction/result.html",
            context
        )

    # ========================================================
    # SHOW PATIENT FORM
    # ========================================================

    return render(
        request,
        "prediction/home.html"
    )


# ============================================================
# USER REGISTRATION
# ============================================================

def register(request):

    if request.method == "POST":

        username = request.POST.get(
            "username",
            ""
        ).strip()

        email = request.POST.get(
            "email",
            ""
        ).strip()

        password = request.POST.get(
            "password",
            ""
        )

        confirm_password = request.POST.get(
            "confirm_password",
            ""
        )

        # =========================
        # BASIC VALIDATION
        # =========================

        if not username or not email or not password:

            return render(
                request,
                "prediction/register.html",
                {
                    "error": "Please fill all fields."
                }
            )

        if len(username) > 150:

            return render(
                request,
                "prediction/register.html",
                {
                    "error": "Username is too long."
                }
            )

        # =========================
        # PASSWORD CONFIRMATION
        # =========================

        if password != confirm_password:

            return render(
                request,
                "prediction/register.html",
                {
                    "error": "Passwords do not match."
                }
            )

        # =========================
        # STRONG PASSWORD VALIDATION
        # =========================

        try:

            validate_password(
                password,
                user=User(username=username, email=email)
            )

        except ValidationError as e:

            return render(
                request,
                "prediction/register.html",
                {
                    "error": e.messages[0]
                }
            )

        # =========================
        # CHECK USERNAME
        # =========================

        if User.objects.filter(
            username=username
        ).exists():

            return render(
                request,
                "prediction/register.html",
                {
                    "error": "Username already exists."
                }
            )

        # =========================
        # CREATE USER
        # =========================

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )

        # =========================
        # LOGIN AUTOMATICALLY
        # =========================

        login(request, user)

        return redirect("prediction")

    # =========================
    # SHOW REGISTRATION PAGE
    # =========================

    return render(
        request,
        "prediction/register.html"
    )


# ============================================================
# USER LOGIN
# ============================================================

def user_login(request):

    if request.method == "POST":

        username = request.POST.get(
            "username",
            ""
        ).strip()

        password = request.POST.get(
            "password",
            ""
        )

        user = authenticate(

            request,

            username=username,

            password=password
        )

        if user is not None:

            login(request, user)

            return redirect("prediction")

        else:

            return render(
                request,
                "prediction/login.html",
                {
                    "error": "Invalid username or password."
                }
            )

    return render(
        request,
        "prediction/login.html"
    )


# ============================================================
# USER LOGOUT
# ============================================================

def user_logout(request):

    logout(request)

    return redirect("login")


# ============================================================
# FEEDBACK
# ============================================================

@login_required(login_url="login")
def feedback(request):

    if request.method == "POST":

        name = request.POST.get(
            "name",
            ""
        ).strip()

        message = request.POST.get(
            "message",
            ""
        ).strip()

        # ----------------------------------------------------
        # Validate feedback
        # ----------------------------------------------------

        if not name or not message:

            return render(
                request,
                "prediction/feedback.html",
                {
                    "error": "Please fill all fields."
                }
            )

        if len(name) > 100:

            return render(
                request,
                "prediction/feedback.html",
                {
                    "error": "Name must be 100 characters or less."
                }
            )

        # ----------------------------------------------------
        # Save feedback
        # ----------------------------------------------------

        Feedback.objects.create(

            name=name,

            message=message
        )

        return render(
            request,
            "prediction/feedback.html",
            {
                "success": "Thank you for your feedback!"
            }
        )

    return render(
        request,
        "prediction/feedback.html"
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

@login_required(login_url="login")
def prediction_history(request):

    predictions = Prediction.objects.filter(

        user=request.user

    ).order_by("-created_at")

    return render(

        request,

        "prediction/history.html",

        {
            "predictions": predictions
        }
    )


# ============================================================
# PREDICTION DETAILS
# ============================================================

@login_required(login_url="login")
def prediction_detail(
    request,
    prediction_id
):

    prediction = get_object_or_404(

        Prediction,

        id=prediction_id,

        user=request.user
    )

    return render(

        request,

        "prediction/detail.html",

        {
            "prediction": prediction
        }
    )


# ============================================================
# DELETE PREDICTION
# ============================================================

@login_required(login_url="login")
def delete_prediction(
    request,
    prediction_id
):

    prediction = get_object_or_404(

        Prediction,

        id=prediction_id,

        user=request.user
    )

    if request.method == "POST":

        prediction.delete()

        return redirect(
            "history"
        )

    return render(

        request,

        "prediction/delete.html",

        {
            "prediction": prediction
        }
    )