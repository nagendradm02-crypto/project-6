# prediction_results.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv("data/processed/cleaned_vehicle.csv")

# Features and target
X = df.drop("Traffic_Density", axis=1)
y = df["Traffic_Density"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("trained_model.pkl")

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("R² Score:", r2)
print("MAE:", mae)

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))

plt.plot(
    range(len(y_test)),
    y_test.values,
    label="Actual"
)

plt.plot(
    range(len(y_pred)),
    y_pred,
    label="Predicted"
)

plt.title("Actual vs Predicted Traffic Density")
plt.xlabel("Sample Index")
plt.ylabel("Traffic Density")
plt.legend()
plt.grid(True)

# Save figure
plt.savefig(
    "prediction_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("prediction_results.png saved successfully!")
