"""Classification module for AirAware AI Air Quality Category prediction.
Trains and compares Decision Tree Classifier and Random Forest Classifier.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.air_quality import CPCB_ORDER


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str]
) -> Dict[str, Any]:
    """Calculate accuracy, macro precision, recall, F1, and confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "Accuracy": float(acc),
        "Precision_Macro": float(prec),
        "Recall_Macro": float(rec),
        "F1_Macro": float(f1),
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }


def train_and_evaluate_classification(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    labels: List[str] = CPCB_ORDER,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train Decision Tree and Random Forest Classifiers on derived air quality categories.
    Evaluates both on the held-out test split.
    """
    # Filter labels that are present in either train or test
    present_labels = [label for label in labels if (label in y_train or label in y_test)]

    # 1. Decision Tree Classifier
    dt = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state
    )
    dt.fit(X_train_scaled, y_train)

    dt_train_preds = dt.predict(X_train_scaled)
    dt_test_preds = dt.predict(X_test_scaled)

    dt_train_metrics = calculate_classification_metrics(y_train, dt_train_preds, present_labels)
    dt_test_metrics = calculate_classification_metrics(y_test, dt_test_preds, present_labels)

    # 2. Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    rf_clf.fit(X_train_scaled, y_train)

    rf_train_preds = rf_clf.predict(X_train_scaled)
    rf_test_preds = rf_clf.predict(X_test_scaled)

    rf_train_metrics = calculate_classification_metrics(y_train, rf_train_preds, present_labels)
    rf_test_metrics = calculate_classification_metrics(y_test, rf_test_preds, present_labels)

    # Model selection based on test Macro F1-score
    if rf_test_metrics["F1_Macro"] >= dt_test_metrics["F1_Macro"]:
        best_clf_name = "Random Forest Classifier"
        best_clf = rf_clf
        best_test_preds = rf_test_preds
    else:
        best_clf_name = "Decision Tree Classifier"
        best_clf = dt
        best_test_preds = dt_test_preds

    return {
        "models": {
            "Decision Tree Classifier": dt,
            "Random Forest Classifier": rf_clf
        },
        "best_model_name": best_clf_name,
        "best_model": best_clf,
        "metrics": {
            "Decision Tree Classifier": {
                "train": dt_train_metrics,
                "test": dt_test_metrics,
                "test_predictions": dt_test_preds
            },
            "Random Forest Classifier": {
                "train": rf_train_metrics,
                "test": rf_test_metrics,
                "test_predictions": rf_test_preds
            }
        },
        "labels": present_labels,
        "y_test": y_test,
        "best_test_predictions": best_test_preds
    }

