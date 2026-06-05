# evaluate.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------
# Load Dataset
# ---------------------------
input_file = "../data/processed/featured_vehicle.csv"

print("Loading dataset...")
df = pd.read_csv(input_file)

# ---------------------------
# Define Target Column
# ---------------------------
target_column = "traffic_volume"

if target_column not in df.columns:
    raise ValueError(
        f"Target column '{target_column}' not found in dataset."
    )

# ---------------------------
# Prepare Features and Target
# ---------------------------
X = df.drop(columns=[target_column])
y = df[target_column]

# Convert categorical variables
X = pd.get_dummies(X, drop_first=True)

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------
# Load Trained Model
# ---------------------------
model_path = "../models/traffic_prediction_model.pkl"

print("Loading trained model...")
model = joblib.load(model_path)

# ---------------------------
# Make Predictions
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# Evaluation Metrics
# ---------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ---------------------------
# Print Results
# ---------------------------
print("\nModel Evaluation Results")
print("=" * 35)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# ---------------------------
# Compare Actual vs Predicted
# ---------------------------
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted Values")
print(comparison.head(10))

# ---------------------------
# Save Evaluation Results
# ---------------------------
comparison.to_csv(
    "../data/processed/evaluation_results.csv",
    index=False
)

print("\nEvaluation results saved successfully!")
print("File: ../data/processed/evaluation_results.csv")
