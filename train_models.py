"""Training script for AirAware AI models.
Executes end-to-end data loading, feature engineering, chronological train/test splitting,
model training for both regression and classification, and saves trained artifacts.
"""

import os
import sys

# Ensure root directory is on Python search path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import joblib
import pandas as pd
import numpy as np

from src.data_loader import load_air_quality_data
from src.preprocessing import (
    create_lag_features,
    chronological_train_test_split,
    AirQualityPreprocessor,
    FEATURE_COLUMNS
)
from src.regression import train_and_evaluate_regression
from src.classification import train_and_evaluate_classification


def run_training():
    print("=" * 70)
    print(" AirAware AI — Model Training Pipeline ")
    print(" AICTE-BharatCares-IBM SkillsBuild Applied AI Project ")
    print("=" * 70)

    # 1. Load Data
    data_path = os.path.join("data", "air_quality_data.csv")
    print(f"\n[1/6] Loading historical air quality data from: {data_path}...")
    df_clean, metadata = load_air_quality_data(data_path)
    print(f"      Loaded {metadata['clean_records']} valid historical records.")
    print(f"      Time period: {metadata['start_date']} to {metadata['end_date']}")

    # 2. Feature Engineering & Lags
    print("\n[2/6] Engineering time-lagged pollutant features & calendar attributes...")
    print("      Forecasting horizon: Next Step (t+1) PM2.5 concentration.")
    df_feat = create_lag_features(df_clean)
    print(f"      Constructed {len(FEATURE_COLUMNS)} feature predictors for {len(df_feat)} target samples.")

    # 3. Chronological Train / Test Split
    print("\n[3/6] Applying chronological train/test split (80% Train, 20% Held-out Test)...")
    train_df, test_df, split_info = chronological_train_test_split(df_feat, train_ratio=0.80)
    print(f"      Train set: {split_info['train_records']} records ({split_info['train_start_date'][:10]} to {split_info['train_end_date'][:10]})")
    print(f"      Test set:  {split_info['test_records']} records ({split_info['test_start_date'][:10]} to {split_info['test_end_date'][:10]})")

    # 4. Fit Preprocessor exclusively on training data
    print("\n[4/6] Fitting median imputer and standard scaler on training split...")
    preprocessor = AirQualityPreprocessor(FEATURE_COLUMNS)
    preprocessor.fit(train_df)

    X_train_scaled = preprocessor.transform(train_df)
    X_test_scaled = preprocessor.transform(test_df)

    y_train_reg = train_df["target_pm25"].values
    y_test_reg = test_df["target_pm25"].values

    y_train_clf = train_df["target_category"].values
    y_test_clf = test_df["target_category"].values

    # 5. Regression Training & Comparison
    print("\n[5/6] Training & evaluating regression models (Predicting PM2.5 µg/m³)...")
    reg_results = train_and_evaluate_regression(
        X_train_scaled, y_train_reg, X_test_scaled, y_test_reg, random_state=42
    )

    print("\n      --- Regression Performance Comparison ---")
    print(f"      {'Model':<26} | {'Test MAE':<10} | {'Test RMSE':<10} | {'Test R²':<10}")
    print("      " + "-" * 62)
    for model_name, res in reg_results["metrics"].items():
        m = res["test"]
        print(f"      {model_name:<26} | {m['MAE']:<10.2f} | {m['RMSE']:<10.2f} | {m['R2']:<10.4f}")
    print(f"\n      => Selected Best Regressor: {reg_results['best_model_name']}")

    # 6. Classification Training & Comparison
    print("\n[6/6] Training & evaluating classification models (CPCB Air Quality Bands)...")
    clf_results = train_and_evaluate_classification(
        X_train_scaled, y_train_clf, X_test_scaled, y_test_clf, random_state=42
    )

    print("\n      --- Classification Performance Comparison ---")
    print(f"      {'Model':<28} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'Macro F1':<10}")
    print("      " + "-" * 76)
    for model_name, res in clf_results["metrics"].items():
        m = res["test"]
        print(f"      {model_name:<28} | {m['Accuracy']:<10.4f} | {m['Precision_Macro']:<10.4f} | {m['Recall_Macro']:<10.4f} | {m['F1_Macro']:<10.4f}")
    print(f"\n      => Selected Best Classifier: {clf_results['best_model_name']}")

    # 7. Save Models and Artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, os.path.join("models", "preprocessor.joblib"))
    joblib.dump(reg_results["best_model"], os.path.join("models", "best_regressor.joblib"))
    joblib.dump(reg_results["models"], os.path.join("models", "all_regressors.joblib"))
    joblib.dump(clf_results["best_model"], os.path.join("models", "best_classifier.joblib"))
    joblib.dump(clf_results["models"], os.path.join("models", "all_classifiers.joblib"))

    summary_artifact = {
        "metadata": metadata,
        "split_info": split_info,
        "feature_columns": FEATURE_COLUMNS,
        "regression": {
            "best_model_name": reg_results["best_model_name"],
            "metrics": reg_results["metrics"],
            "test_actuals": y_test_reg,
            "best_predictions": reg_results["best_test_predictions"],
            "test_timestamps": test_df["target_timestamp"].astype(str).tolist()
        },
        "classification": {
            "best_model_name": clf_results["best_model_name"],
            "metrics": clf_results["metrics"],
            "labels": clf_results["labels"],
            "test_actuals": y_test_clf.tolist(),
            "best_predictions": clf_results["best_test_predictions"].tolist()
        }
    }
    joblib.dump(summary_artifact, os.path.join("models", "training_summary.joblib"))

    print("\n" + "=" * 70)
    print(" All models and artifacts successfully saved to ./models/ directory!")
    print("=" * 70)


if __name__ == "__main__":
    run_training()
