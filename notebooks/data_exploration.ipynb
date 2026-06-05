# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('cleaned_vehicle.csv')

# Display first rows
print(df.head())

# Dataset shape
print("Shape:", df.shape)

# Column names
print("Columns:")
print(df.columns)

# Dataset information
print(df.info())

# Summary statistics
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Check duplicate rows
print("Duplicates:", df.duplicated().sum())

# Correlation matrix (numerical columns)
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Histograms
numeric_df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Boxplots for numerical columns
for col in numeric_df.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Value counts for categorical columns
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    print(f"\n{col}")
    print(df[col].value_counts())

# Pairplot (first few numerical columns)
if len(numeric_df.columns) > 1:
    sns.pairplot(df[numeric_df.columns[:5]])
    plt.show()

print("Data Exploration Completed")
