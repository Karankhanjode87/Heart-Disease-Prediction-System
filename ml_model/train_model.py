import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# Load dataset
df = pd.read_csv("ml_model/heart_disease_dashboard.csv")


# Separate input features (X) and target (y)
X = df.drop(columns=["Heart_Disease", "Age_Group"])
y = df["Heart_Disease"]


# Define categorical and numerical features
categorical_features = [
    "Gender",
    "Smoking",
    "Alcohol_Intake",
    "Physical_Activity",
    "Diet",
    "Stress_Level"
]

numerical_features = [
    "Age",
    "Weight",
    "Height",
    "BMI",
    "Hypertension",
    "Diabetes",
    "Hyperlipidemia",
    "Family_History",
    "Previous_Heart_Attack",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart_Rate",
    "Blood_Sugar_Fasting",
    "Cholesterol_Total"
]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            StandardScaler(),
            numerical_features
        )
    ]
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]
)


# Train the model
model.fit(X_train, y_train)
print("Model training completed!")
joblib.dump(model, "ml_model/heart_disease_model.pkl")

print("Model saved successfully!")
# Make predictions on unseen test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
rf_model = model.named_steps["classifier"]

feature_names = model.named_steps["preprocessor"].get_feature_names_out()

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nTop 15 Important Features:")
print(feature_importance.head(15).to_string(index=False))