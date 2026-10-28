"""Evaluation script for AirAware AI models.
Loads trained models and held-out test evaluation artifacts to generate comprehensive performance reports.
"""

import os
import sys

# Ensure root directory is on Python search path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import joblib
import numpy as np
import pandas as pd


def run_evaluation():
    summary_path = os.path.join("models", "training_summary.joblib")
    if not os.path.exists(summary_path):
        print(f"Error: Model artifacts not found at '{summary_path}'.")
        print("Please train the models first by running: python train_models.py")
        sys.exit(1)

    summary = joblib.load(summary_path)

    split = summary["split_info"]
    meta = summary["metadata"]
    reg = summary["regression"]
    clf = summary["classification"]

    print("=" * 75)
    print(" AirAware AI — Model Evaluation & Performance Report ")
    print(" Aligned with UN SDG 3 (Good Health) & SDG 11 (Sustainable Cities) ")
    print("=" * 75)

    print("\n1. DATASET & CHRONOLOGICAL PARTITION")
    print("-" * 50)
    print(f"Total Valid Records:       {split['total_records']}")
    print(f"Training Set (Earliest):   {split['train_records']} records ({split['train_start_date'][:10]} to {split['train_end_date'][:10]})")
    print(f"Testing Set (Held-Out):    {split['test_records']} records ({split['test_start_date'][:10]} to {split['test_end_date'][:10]})")
    print(f"Split Methodology:         Strict Chronological Split (No data leakage, no lookahead)")

    print("\n2. REGRESSION MODELS (PM2.5 Numerical Concentration Forecast in µg/m³)")
    print("-" * 75)
    print(f"{'Model Name':<28} | {'Split':<6} | {'MAE (µg/m³)':<13} | {'RMSE (µg/m³)':<13} | {'R²':<8}")
    print("-" * 75)

    for name, data in reg["metrics"].items():
        tr = data["train"]
        te = data["test"]
        print(f"{name:<28} | {'Train':<6} | {tr['MAE']:<13.2f} | {tr['RMSE']:<13.2f} | {tr['R2']:<8.4f}")
        print(f"{'':<28} | {'Test':<6} | {te['MAE']:<13.2f} | {te['RMSE']:<13.2f} | {te['R2']:<8.4f}")
        print("-" * 75)

    print(f"Selected Best Regressor: {reg['best_model_name']}")

    actuals = np.array(reg["test_actuals"])
    preds = np.array(reg["best_predictions"])
    residuals = preds - actuals
    print(f"Mean Residual (Bias):     {np.mean(residuals):+.2f} µg/m³")
    print(f"Residual Std Deviation:   {np.std(residuals):.2f} µg/m³")
    print(f"Mean Actual PM2.5 (Test): {np.mean(actuals):.2f} µg/m³")

    print("\n3. CLASSIFICATION MODELS (CPCB Air Quality Category Prediction)")
    print("-" * 75)
    print(f"{'Classifier':<28} | {'Accuracy':<10} | {'Macro Prec':<12} | {'Macro Rec':<11} | {'Macro F1':<10}")
    print("-" * 75)

    for name, data in clf["metrics"].items():
        te = data["test"]
        print(f"{name:<28} | {te['Accuracy']:<10.4f} | {te['Precision_Macro']:<12.4f} | {te['Recall_Macro']:<11.4f} | {te['F1_Macro']:<10.4f}")

    print("-" * 75)
    print(f"Selected Best Classifier: {clf['best_model_name']}")

    best_clf_name = clf["best_model_name"]
    cm = clf["metrics"][best_clf_name]["test"]["confusion_matrix"]
    labels = clf["labels"]

    print(f"\n4. CONFUSION MATRIX ({best_clf_name})")
    print("-" * 65)
    header = f"{'Actual \\ Predicted':<22} | " + " | ".join([f"{l[:8]:>8}" for l in labels])
    print(header)
    print("-" * len(header))
    for idx, true_lbl in enumerate(labels):
        row_str = f"{true_lbl[:20]:<22} | " + " | ".join([f"{cm[idx][j]:>8}" for j in range(len(labels))])
        print(row_str)

    print("\n" + "=" * 75)
    print(" Evaluation complete. All metrics reflect genuine held-out chronological test data.")
    print("=" * 75)


if __name__ == "__main__":
    run_evaluation()
