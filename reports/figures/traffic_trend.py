# traffic_trend.py

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/processed/cleaned_vehicle.csv")

# Create figure
plt.figure(figsize=(12, 6))

# Plot traffic trend
plt.plot(df.index, df["Traffic_Density"], marker="o")

# Labels and title
plt.title("Traffic Density Trend")
plt.xlabel("Record Number")
plt.ylabel("Traffic Density")

# Grid
plt.grid(True)

# Save plot
plt.savefig("traffic_trend.png", dpi=300, bbox_inches="tight")

# Display plot
plt.show()

print("traffic_trend.png saved successfully!")
