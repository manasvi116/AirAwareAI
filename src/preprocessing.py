"""Data preprocessing and leakage-free feature engineering for AirAware AI.

Forecasting Horizon:
- Next-step (t+1) PM2.5 forecasting.
- Given historical measurements available at or before observation t,
  the models predict the PM2.5 concentration at observation t+1.
- All lag and rolling features strictly use data up to time t.
- Transformations (imputation, scaling) are fitted ONLY on the chronological training split.
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.air_quality import categorize_pm25

FEATURE_COLUMNS = [
    "pm25_lag1",
    "pm25_lag2",
    "pm25_lag3",
    "pm25_lag7",
    "pm25_roll_mean_7",
    "pm10_lag1",
    "no2_lag1",
    "so2_lag1",
    "co_lag1",
    "o3_lag1",
    "month",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "sin_month",
    "cos_month"
]


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct strictly backwards-looking lag and rolling features.

    Target: pm25 at time t+1.
    Features: measurements available up to time t, plus time-calendar properties of t+1.
    """
    df_feat = df.copy()

    # Ensure chronological order
    df_feat = df_feat.sort_values("timestamp").reset_index(drop=True)

    # Historical PM2.5 lags at time t, t-1, t-2, t-6
    df_feat["pm25_lag1"] = df_feat["pm25"].shift(0)
    df_feat["pm25_lag2"] = df_feat["pm25"].shift(1)
    df_feat["pm25_lag3"] = df_feat["pm25"].shift(2)
    df_feat["pm25_lag7"] = df_feat["pm25"].shift(6)

    # 7-day rolling mean up to time t (shift=0 on historical PM2.5 series)
    df_feat["pm25_roll_mean_7"] = df_feat["pm25"].rolling(window=7, min_periods=1).mean()

    # Other pollutant measurements available at time t
    for col in ["pm10", "no2", "so2", "co", "o3"]:
        if col in df_feat.columns:
            df_feat[f"{col}_lag1"] = df_feat[col].shift(0)
        else:
            # If a pollutant column does not exist, initialize with NaN for imputer
            df_feat[f"{col}_lag1"] = np.nan

    # Target: PM2.5 at next step (t+1)
    df_feat["target_pm25"] = df_feat["pm25"].shift(-1)
    df_feat["target_timestamp"] = df_feat["timestamp"].shift(-1)

    # Calendar features valid at the forecasting horizon target timestamp (t+1)
    target_dt = pd.to_datetime(df_feat["target_timestamp"])
    df_feat["month"] = target_dt.dt.month
    df_feat["day_of_week"] = target_dt.dt.dayofweek
    df_feat["day_of_year"] = target_dt.dt.dayofyear
    df_feat["is_weekend"] = target_dt.dt.dayofweek.isin([5, 6]).astype(int)

    # Cyclical month encodings
    df_feat["sin_month"] = np.sin(2 * np.pi * df_feat["month"] / 12.0)
    df_feat["cos_month"] = np.cos(2 * np.pi * df_feat["month"] / 12.0)

    # Drop rows without target (the last row) and initial lag warm-up rows where pm25_lag7 is NaN
    # We require at least 7 historical observations to construct reliable lag features
    df_feat = df_feat.dropna(subset=["target_pm25"]).copy()
    df_feat = df_feat.iloc[6:].reset_index(drop=True)

    # Derive classification target category using documented CPCB thresholds
    df_feat["target_category"] = df_feat["target_pm25"].apply(
        lambda v: categorize_pm25(v, standard="CPCB")["category"]
    )

    return df_feat


def chronological_train_test_split(
    df_feat: pd.DataFrame,
    train_ratio: float = 0.80
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split the dataset chronologically without shuffling.
    Earliest observations form the training set; subsequent observations form the test set.
    """
    n_total = len(df_feat)
    split_idx = int(n_total * train_ratio)

    train_df = df_feat.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_feat.iloc[split_idx:].copy().reset_index(drop=True)

    split_info = {
        "total_records": n_total,
        "train_records": len(train_df),
        "test_records": len(test_df),
        "train_ratio": train_ratio,
        "train_start_date": str(train_df["timestamp"].min()),
        "train_end_date": str(train_df["timestamp"].max()),
        "test_start_date": str(test_df["timestamp"].min()),
        "test_end_date": str(test_df["timestamp"].max()),
    }

    return train_df, test_df, split_info


class AirQualityPreprocessor:
    """
    Preprocessing pipeline fitted exclusively on chronological training data.
    Encapsulates median imputation and feature scaling.
    """

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_medians: Dict[str, float] = {}

    def fit(self, X_train: pd.DataFrame) -> "AirQualityPreprocessor":
        """Fit imputer and scaler only on training data."""
        X_sub = X_train[self.feature_columns].copy()
        imputed_array = self.imputer.fit_transform(X_sub)
        self.scaler.fit(imputed_array)

        # Store training feature medians for single-record inference fallback
        for idx, col in enumerate(self.feature_columns):
            self.feature_medians[col] = float(self.imputer.statistics_[idx])

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted training statistics."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        X_sub = X[self.feature_columns].copy()
        imputed = self.imputer.transform(X_sub)
        scaled = self.scaler.transform(imputed)
        return scaled

    def transform_single_input(self, input_dict: Dict[str, Any]) -> np.ndarray:
        """Transform a single observation dictionary for real-time web inference."""
        row_dict = {}
        for col in self.feature_columns:
            val = input_dict.get(col, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                row_dict[col] = self.feature_medians.get(col, 0.0)
            else:
                row_dict[col] = float(val)

        df_single = pd.DataFrame([row_dict], columns=self.feature_columns)
        return self.transform(df_single)

