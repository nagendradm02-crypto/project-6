# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import joblib

# Load feature-engineered dataset
input_file = "../data/processed/featured_vehicle.csv"

print("Loading dataset...")
df = pd.read_csv(input_file)

print("\nDataset Shape:", df.shape)

# Define target column
target_column = "traffic_volume"

if target_column not in df.columns:
    raise ValueError(
        f"Target column '{target_column}' not found in dataset."
    )

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Convert categorical columns to numerical
X = pd.get_dummies(X, drop_first=True)

print("\nFeature Matrix Shape:", X.shape)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# Train Model
print("\nTraining Random Forest Model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-" * 30)
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 4))

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "
