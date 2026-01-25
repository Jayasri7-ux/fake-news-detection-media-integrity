import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.models.model_trainer import ModelTrainer
from src.models.model_calibration import ModelCalibrator
from src.models.threshold_optimizer import ThresholdOptimizer
from src.models.model_comparison import ModelComparator

def main():
    print("Starting Advanced Model Optimization Pipeline...")
    
    # 1. Load Data
    data_loader = DataLoader() # Assuming data is in 'data/processed' relative to root or we pass full paths
    base_path = r"c:\project\fake_news_detection_media_integrity\data\processed"
    
    # Use pickle loading as per our update
    print("Loading data...")
    try:
        X_train = data_loader.load_pickle(os.path.join(base_path, "X_train.pkl"))
        y_train = data_loader.load_pickle(os.path.join(base_path, "y_train.pkl"))
        X_test = data_loader.load_pickle(os.path.join(base_path, "X_test.pkl"))
        y_test = data_loader.load_pickle(os.path.join(base_path, "y_test.pkl"))
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Prepare Data Splits
    # Split Train -> Train (for tuning) / Val (for calibration & thresholding)
    print("Splitting training data for proper calibration...")
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 3. Hyperparameter Tuning
    # Initialize trainer with the optimization fold
    trainer = ModelTrainer(X_train_opt, y_train_opt)
    trained_models = {}

    # Logistic Regression
    print("\nTuning Logistic Regression...")
    lr_params = {'C': [0.1, 1, 10], 'solver': ['liblinear']}
    lr_best = trainer.train(LogisticRegression(max_iter=1000), lr_params)
    trained_models['LogisticRegression'] = lr_best

    # Random Forest
    print("\nTuning Random Forest...")
    rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10]} # Reduced for speed in demo
    rf_best = trainer.train(RandomForestClassifier(random_state=42), rf_params)
    trained_models['RandomForest'] = rf_best
    
    # Gradient Boosting
    print("\nTuning Gradient Boosting...")
    gb_params = {'n_estimators': [50], 'learning_rate': [0.1]} # Reduced for speed
    gb_best = trainer.train(GradientBoostingClassifier(random_state=42), gb_params)
    trained_models['GradientBoosting'] = gb_best

    # 4. Model Calibration
    print("\nCalibrating Models (using CV on training set)...")
    calibrator = ModelCalibrator()
    
    # Calibrate the best RF model
    # We pass the optimized training set. The CalibratedClassifierCV will handle internal splitting.
    rf_calibrated = calibrator.calibrate(rf_best, X_train_opt, y_train_opt)
    trained_models['RandomForest_Calibrated'] = rf_calibrated

    # 5. Threshold Optimization
    print("\nOptimizing Thresholds (using validation set)...")
    optimizer = ThresholdOptimizer()
    rf_best_thresh = optimizer.optimize_threshold(rf_best, X_val, y_val, metric='f1')
    print(f"Optimal Threshold for Random Forest (F1): {rf_best_thresh}")
    
    rf_biz_thresh = optimizer.optimize_for_business(rf_best, X_val, y_val, cost_fp=1, cost_fn=5)
    print(f"Optimal Threshold for Random Forest (Business Cost): {rf_biz_thresh}")

    # 5. Model Comparison & Reporting
    print("\nGenerating Comparison Report...")
    comparator = ModelComparator()
    results_df = comparator.compare_models(trained_models, X_test, y_test)
    print(results_df)
    
    comparator.generate_report(results_df, output_path=r"c:\project\fake_news_detection_media_integrity\model_comparison_report.md")
    
    print("\nPipeline Complete!")

if __name__ == "__main__":
    main()
