# data_preprocessing.py

import pandas as pd
import numpy as np

# Load raw dataset
input_file = "../data/raw/Vehicle.csv"
output_file = "../data/processed/cleaned_vehicle.csv"

print("Loading dataset...")
df = pd.read_csv(input_file)

# Display basic information
print("\nOriginal Shape:", df.shape)
print(df.head())

# Remove duplicate rows
duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

df = df.drop_duplicates()

# Handle missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Numerical columns
num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Remove extra spaces from text columns
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip()

# Convert column names to lowercase
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Remove rows with invalid numerical values
for col in num_cols:
    df = df[df[col].notna()]

# Reset index
df.reset_index(drop=True, inplace=True)

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("\nData Preprocessing Completed Successfully!")
print("Cleaned Shape:", df.shape)
print(f"Cleaned dataset saved to: {output_file}")
