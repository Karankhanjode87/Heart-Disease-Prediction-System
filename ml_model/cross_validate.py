import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("ml_model/heart_disease_dashboard.csv")

X = df.drop(columns=["Heart_Disease", "Age_Group"])
y = df["Heart_Disease"]


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


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ]
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ]
)


cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


print("Cross-validation scores:")

for i, score in enumerate(scores, 1):
    print(f"Fold {i}: {score:.4f}")

print(f"\nMean accuracy: {scores.mean():.4f}")