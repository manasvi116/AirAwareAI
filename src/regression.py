"""Regression module for AirAware AI PM2.5 numerical concentration forecasting.
Trains and compares Linear Regression and Random Forest Regressor.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, and R2 regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


def train_and_evaluate_regression(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train Linear Regression and Random Forest Regressor on scaled training data.
    Evaluate on both train and test splits to detect overfitting and assess generalization.
    """
    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    lr_train_preds = lr.predict(X_train_scaled)
    # Clip negative predictions since PM2.5 concentration cannot be negative physically
    lr_train_preds = np.clip(lr_train_preds, a_min=0.0, a_max=None)
    lr_test_preds = lr.predict(X_test_scaled)
    lr_test_preds = np.clip(lr_test_preds, a_min=0.0, a_max=None)

    lr_train_metrics = calculate_regression_metrics(y_train, lr_train_preds)
    lr_test_metrics = calculate_regression_metrics(y_test, lr_test_preds)

    # 2. Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)

    rf_train_preds = rf.predict(X_train_scaled)
    rf_train_preds = np.clip(rf_train_preds, a_min=0.0, a_max=None)
    rf_test_preds = rf.predict(X_test_scaled)
    rf_test_preds = np.clip(rf_test_preds, a_min=0.0, a_max=None)

    rf_train_metrics = calculate_regression_metrics(y_train, rf_train_preds)
    rf_test_metrics = calculate_regression_metrics(y_test, rf_test_preds)

    # Model selection based on test set RMSE
    if rf_test_metrics["RMSE"] <= lr_test_metrics["RMSE"]:
        best_model_name = "Random Forest Regressor"
        best_model = rf
        best_test_preds = rf_test_preds
    else:
        best_model_name = "Linear Regression"
        best_model = lr
        best_test_preds = lr_test_preds

    return {
        "models": {
            "Linear Regression": lr,
            "Random Forest Regressor": rf
        },
        "best_model_name": best_model_name,
        "best_model": best_model,
        "metrics": {
            "Linear Regression": {
                "train": lr_train_metrics,
                "test": lr_test_metrics,
                "test_predictions": lr_test_preds
            },
            "Random Forest Regressor": {
                "train": rf_train_metrics,
                "test": rf_test_metrics,
                "test_predictions": rf_test_preds
            }
        },
        "y_test": y_test,
        "best_test_predictions": best_test_preds
    }

