# main.py

import os
import subprocess


def run_script(script_path):
    """
    Execute a Python script and display status.
    """
    print(f"\nRunning {script_path}...")

    try:
        subprocess.run(
            ["python", script_path],
            check=True
        )
        print(f"{script_path} completed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_path}")
        print(e)


def main():

    print("=" * 50)
    print("VEHICLE TRAFFIC ANALYSIS PROJECT")
    print("=" * 50)

    scripts = [
        "src/data_preprocessing.py",
        "src/feature_engineering.py",
        "src/train_model.py",
        "src/evaluate.py",
        "project_report.py"
    ]

    for script in scripts:

        if os.path.exists(script):
            run_script(script)
        else:
            print(f"File not found: {script}")

    print("\n" + "=" * 50)
    print("PROJECT EXECUTION COMPLETED")
    print("=" * 50)

    print("\nGenerated Files:")
    print("- cleaned_vehicle.csv")
    print("- trained_model.pkl")
    print("- scaler.pkl")
    print("- traffic_trend.png")
    print("- prediction_results.png")
    print("- project_report.pdf")


if __name__ == "__main__":
    main()
