import pandas as pd

df = pd.read_csv("ml_model/heart_disease_dashboard.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

print("\nTarget Distribution:")
print(df["Heart_Disease"].value_counts())