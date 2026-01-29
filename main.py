import os
import sys
import time


def run_command(command, step_name):
    print("\n" + "=" * 60)
    print(f"RUNNING: {step_name}")
    print("=" * 60)

    start = time.time()
    exit_code = os.system(command)
    duration = time.time() - start

    if exit_code != 0:
        print(f"\n[ERROR] {step_name} failed")
        sys.exit(1)

    print(f"\n[OK] {step_name} completed in {duration:.2f}s")


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)  # ensure project root execution

    print("\n" + "*" * 60)
    print("FAKE NEWS DETECTION – FULL PIPELINE (TERMINAL MODE)")
    print("*" * 60)
    print(f"Project Root: {BASE_DIR}")

    steps = [
        ("01 Data Collection", "python extracted_scripts/01_data_collection.py"),
        ("02 Data Cleaning", "python extracted_scripts/02_data_cleaning.py"),
        ("03 Data Quality Report", "python extracted_scripts/03_data_quality_report.py"),
        ("04 EDA", "python extracted_scripts/04_eda.py"),
        ("05 Feature Engineering", "python extracted_scripts/05_feature_engineering.py"),
        ("06 Model Training", "python extracted_scripts/06_model_training.py"),
        ("07 Model Evaluation", "python extracted_scripts/07_model_evaluation.py"),
        ("08 Model Comparison", "python extracted_scripts/08_model_comparison.py"),
        ("09 Training Comparison", "python extracted_scripts/09_model_training_comparison.py"),
        ("10 EDA Dashboard", "python extracted_scripts/10_eda_dashboard.py"),
    ]

    pipeline_start = time.time()

    for name, cmd in steps:
        run_command(cmd, name)

    print("\n" + "=" * 60)
    print("PREDICTION (USER INPUT)")
    print("=" * 60)

    run_command("python predict.py", "Prediction")

    total_time = time.time() - pipeline_start

    print("\n" + "*" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY ✅")
    print(f"Total Execution Time: {total_time:.2f}s")
    print("*" * 60)


if __name__ == "__main__":
    main()
