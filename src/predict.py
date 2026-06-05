# predict.py

import pandas as pd
import joblib

# ---------------------------
# Load Trained Model
# ---------------------------
model_path = "../models/traffic_prediction_model.pkl"

print("Loading trained model...")
model = joblib.load(model_path)

# ---------------------------
# Load New Data
# ---------------------------
input_file = "../data/processed/featured_vehicle.csv"

print("Loading input data...")
df = pd.read_csv(input_file)

# ---------------------------
# Prepare Features
# ---------------------------
target_column = "traffic_volume"

if target_column in df.columns:
    X = df.drop(columns=[target_column])
else:
    X = df.copy()

# Convert categorical columns to numerical
X = pd.get_dummies(X, drop_first=True)

# ---------------------------
# Make Predictions
# ---------------------------
print("Making predictions...")

predictions = model.predict(X)

# Store predictions
results = df.copy()
results["Predicted_Traffic_Volume"] = predictions

# ---------------------------
# Save Results
# ---------------------------
output_file = "../data/processed/predictions.csv"

results.to_csv(output_file, index=False)

print("\nPredictions completed successfully!")
print(f"Results saved to: {output_file}")

# Display first 10 predictions
print("\nSample Predictions:")
print(results[["Predicted_Traffic_Volume"]].head(10))
