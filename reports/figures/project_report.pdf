from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet

# Create PDF document
pdf = SimpleDocTemplate("project_report.pdf")

# Styles
styles = getSampleStyleSheet()

title_style = styles["Title"]
heading_style = styles["Heading2"]
body_style = styles["BodyText"]

content = []

# Title Page
content.append(
    Paragraph(
        "Vehicle Traffic Analysis Using Machine Learning",
        title_style
    )
)

content.append(Spacer(1, 20))

content.append(
    Paragraph(
        "Project Report",
        heading_style
    )
)

content.append(
    Paragraph(
        "Prepared By: Nagendra D M",
        body_style
    )
)

content.append(
    Paragraph(
        "Academic Mini Project",
        body_style
    )
)

content.append(PageBreak())

# Introduction
content.append(Paragraph("1. Introduction", heading_style))
content.append(
    Paragraph(
        """
        Vehicle Traffic Analysis is a machine learning project
        developed to analyze traffic-related data and predict
        traffic density. The system helps understand traffic
        patterns and supports data-driven decision making.
        """,
        body_style
    )
)

content.append(Spacer(1, 12))

# Objectives
content.append(Paragraph("2. Objectives", heading_style))
content.append(
    Paragraph(
        """
        • Analyze vehicle traffic datasets.<br/>
        • Perform data preprocessing.<br/>
        • Engineer useful features.<br/>
        • Train a machine learning model.<br/>
        • Predict traffic density.<br/>
        • Evaluate model performance.
        """,
        body_style
    )
)

content.append(Spacer(1, 12))

# Dataset
content.append(Paragraph("3. Dataset Description", heading_style))
content.append(
    Paragraph(
        """
        The dataset contains vehicle-related information such as
        vehicle count, speed, weather conditions, road conditions,
        and traffic density values.
        """,
        body_style
    )
)

content.append(Spacer(1, 12))

# Preprocessing
content.append(Paragraph("4. Data Preprocessing", heading_style))
content.append(
    Paragraph(
        """
        Data preprocessing includes handling missing values,
        removing duplicates, encoding categorical variables,
        and preparing data for machine learning algorithms.
        """,
        body_style
    )
)

content.append(Spacer(1, 12))

# Feature Engineering
content.append(Paragraph("5. Feature Engineering", heading_style))
content.append(
    Paragraph(
        """
        Feature engineering is performed to improve model
        performance by selecting and transforming useful
        features from the dataset.
        """,
        body_style
    )
)

content.append(Spacer(1, 12))

# Model Training
content.append(Paragraph("6. Model Training", heading_style))
content.append(
    Paragraph(
        """
        A Random Forest Regressor model is trained using
        processed traffic data. The trained model is saved
        as trained_model.pkl.
        """,
        body_style
    )
)

content.append(Spacer(1, 12))

# Evaluation
content.append(Paragraph("7. Model Evaluation", heading_style))
content.append(
    Paragraph(
        """
        The model performance is evaluated using:
        <br/>
        • Mean Absolute Error (MAE)
        <br/>
        • R² Score
        """,
        body_style
    )
)

content.append(Spacer(1, 12))

# Visualizations
content.append(Paragraph("8. Visualizations", heading_style))
content.append(
    Paragraph(
        """
        The project generates:
        <br/>
        • traffic_trend.png
        <br/>
        • prediction_results.png
        <br/>
        These visualizations help understand traffic patterns
        and prediction performance.
        """,
        body_style
    )
)

content.append(Spacer(1, 12))

# Outputs
content.append(Paragraph("9. Project Outputs", heading_style))
content.append(
    Paragraph(
        """
        • cleaned_vehicle.csv<br/>
        • trained_model.pkl<br/>
        • scaler.pkl<br/>
        • traffic_trend.png<br/>
        • prediction_results.png<br/>
        • project_report.pdf
        """,
        body_style
    )
)

content.append(Spacer(1, 12))

# Conclusion
content.append(Paragraph("10. Conclusion", heading_style))
content.append(
    Paragraph(
        """
        The Vehicle Traffic Analysis project demonstrates
        the application of machine learning techniques for
        traffic prediction. The system successfully preprocesses
        data, trains a predictive model, evaluates performance,
        and generates useful visualizations.
        """,
        body_style
    )
)

content.append(PageBreak())

# References
content.append(Paragraph("11. References", heading_style))
content.append(
    Paragraph(
        """
        1. Python Documentation<br/>
        2. Pandas Documentation<br/>
        3. NumPy Documentation<br/>
        4. Scikit-Learn Documentation<br/>
        5. Matplotlib Documentation<br/>
        6. ReportLab Documentation
        """,
        body_style
    )
)

# Build PDF
pdf.build(content)

print("project_report.pdf generated successfully!")
