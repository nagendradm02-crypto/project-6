# train_model.py

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================
# Load Dataset
# ==========================

DATA_PATH = "data/processed/cleaned_vehicle.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# ==========================
# Select Features & Target
# ==========================

# Replace 'Vehicles' with your target column name if different
target_column = "Vehicles"

if target_column not in df.columns:
    raise ValueError(
        f"Target column '{target_column}' not found.\n"
        f"Available columns: {list(df.columns)}"
    )

X = df.drop(columns=[target_column])
y = df[target_column]

# ==========================
# Handle Categorical Columns
# ==========================

X = pd.get_dummies(X, drop_first=True)

print("Feature Shape:", X.shape)

# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train Model
# ==========================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed")

# ==========================
# Predictions
# ==========================

y_pred = model.predict(X_test)

# ==========================
# Evaluation
# ==========================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-" * 30)
print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("R²  :", round(r2, 4))

# ==========================
# Save Model
# ==========================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/vehicle_model.pkl")

print("\nModel saved successfully!")
print("Location: models/vehicle_model.pkl")
