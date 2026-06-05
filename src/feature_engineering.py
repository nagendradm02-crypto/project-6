# feature_engineering.py

import pandas as pd
import numpy as np

# Load cleaned dataset
input_file = "../data/processed/cleaned_vehicle.csv"
output_file = "../data/processed/featured_vehicle.csv"

print("Loading cleaned dataset...")
df = pd.read_csv(input_file)

print("\nDataset Shape:", df.shape)

# ---------------------------
# Feature Engineering
# ---------------------------

# Convert datetime columns if present
for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        try:
            df[col] = pd.to_datetime(df[col])

            # Extract useful time-based features
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_hour"] = df[col].dt.hour

        except:
            pass

# Encode categorical variables
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    if df[col].nunique() <= 20:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)

# Create vehicle age feature (if year column exists)
if 'year' in df.columns:
    current_year = pd.Timestamp.now().year
    df['vehicle_age'] = current_year - df['year']

# Create speed category (if speed column exists)
if 'speed' in df.columns:
    df['speed_category'] = pd.cut(
        df['speed'],
        bins=[0, 40, 80, 120, np.inf],
        labels=['Low', 'Medium', 'High', 'Very High']
    )

# Create traffic density category (if traffic_volume exists)
if 'traffic_volume' in df.columns:
    df['traffic_density'] = pd.cut(
        df['traffic_volume'],
        bins=[0, 100, 500, 1000, np.inf],
        labels=['Low', 'Moderate', 'Heavy', 'Very Heavy']
    )

# Remove original datetime columns if desired
datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
df.drop(columns=datetime_cols, inplace=True, errors='ignore')

# Save engineered dataset
df.to_csv(output_file, index=False)

print("\nFeature Engineering Completed Successfully!")
print("New Shape:", df.shape)
print(f"Feature-engineered dataset saved to: {output_file}")

print("\nNew Columns Added:")
for col in df.columns:
    print(col)
