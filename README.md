Vehicle Traffic Analysis Using Machine Learning

Project Overview

The Vehicle Traffic Analysis project uses Machine Learning techniques to analyze traffic data and predict traffic density. The project includes data preprocessing, feature engineering, model training, evaluation, visualization, and report generation.

---

Project Structure

Vehicle-Traffic-Analysis/
│
├── data/
│   ├── raw/
│   │   └── Vehicle.csv
│   └── processed/
│       └── cleaned_vehicle.csv
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   └── evaluate.py
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
│
├── trained_model.pkl
├── scaler.pkl
├── traffic_trend.png
├── prediction_results.png
├── project_report.pdf
├── requirements.txt
└── README.md

---

Features

- Data Cleaning and Preprocessing
- Feature Engineering
- Traffic Density Prediction
- Model Evaluation
- Data Visualization
- Automated PDF Report Generation
- Unit Testing

---

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib
- ReportLab
- Jupyter Notebook

---

Installation

Clone the Repository

git clone <repository-url>
cd Vehicle-Traffic-Analysis

Install Dependencies

pip install -r requirements.txt

---

Dataset

Place the dataset file inside:

data/raw/Vehicle.csv

The dataset should contain traffic-related information such as:

- Vehicle Count
- Speed
- Road Condition
- Weather Condition
- Traffic Density

---

Running the Project

Step 1: Data Preprocessing

python src/data_preprocessing.py

Output:

cleaned_vehicle.csv

---

Step 2: Feature Engineering

python src/feature_engineering.py

---

Step 3: Train the Model

python src/train_model.py

Outputs:

trained_model.pkl
scaler.pkl

---

Step 4: Make Predictions

python src/predict.py

---

Step 5: Evaluate the Model

python src/evaluate.py

Metrics:

- Mean Absolute Error (MAE)
- R² Score

---

Visualizations

Traffic Trend

Generated File:

traffic_trend.png

Shows traffic density trends in the dataset.

Prediction Results

Generated File:

prediction_results.png

Shows comparison between actual and predicted traffic density values.

---

Testing

Test Data Preprocessing

python tests/test_preprocessing.py

Test Model

python tests/test_model.py

---

PDF Report

Generate Project Report:

python project_report.py

Output:

project_report.pdf

The report includes:

- Introduction
- Objectives
- Dataset Description
- Data Preprocessing
- Feature Engineering
- Model Training
- Evaluation
- Visualizations
- Conclusion

---

Requirements

pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
reportlab
jupyter
notebook

Install using:

pip install -r requirements.txt

---

Future Enhancements

- Real-time Traffic Prediction
- Deep Learning Models
- Traffic Congestion Detection
- Interactive Dashboard
- Deployment using Flask or Streamlit

---

Author

Nagendra D M

Vehicle Traffic Analysis Project
Machine Learning Mini Project
