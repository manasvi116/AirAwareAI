"""Unit tests for data loading, lag creation, and chronological split."""

import pytest
import pandas as pd
import numpy as np
import datetime
from src.data_loader import standardize_columns, load_air_quality_data
from src.preprocessing import (
    create_lag_features,
    chronological_train_test_split,
    AirQualityPreprocessor,
    FEATURE_COLUMNS
)


def test_standardize_columns():
    """Verify that common variations in column headers are correctly normalized."""
    df = pd.DataFrame({
        "Date": ["2023-01-01"],
        "PM2.5": [50.0],
        "PM10": [100.0],
        "NO2": [25.0],
        "CO": [1.0]
    })
    standardized = standardize_columns(df)
    assert "timestamp" in standardized.columns
    assert "pm25" in standardized.columns
    assert "pm10" in standardized.columns
    assert "no2" in standardized.columns
    assert "co" in standardized.columns


def test_create_lag_features_leakage_prevention():
    """Verify that lag features use only historical data (t, t-1, ...) and target is strictly t+1."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    # Strictly increasing sequence so values reflect chronological position
    pm25_vals = [float(i * 10) for i in range(20)]

    df = pd.DataFrame({
        "timestamp": dates,
        "pm25": pm25_vals,
        "pm10": [v * 1.5 for v in pm25_vals]
    })

    df_feat = create_lag_features(df)

    # For any row, target_pm25 should be greater than pm25_lag1
    for idx, row in df_feat.iterrows():
        # pm25_lag1 must equal the observation at t, target_pm25 must be observation at t+1
        assert row["target_pm25"] > row["pm25_lag1"]
        # pm25_lag1 > pm25_lag2
        assert row["pm25_lag1"] > row["pm25_lag2"]


def test_chronological_split_strict_order():
    """Verify that training timestamps strictly precede testing timestamps (no leakage)."""
    dates = pd.date_range("2022-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "pm25": np.random.uniform(20, 150, size=50)
    })
    df_feat = create_lag_features(df)

    train_df, test_df, split_info = chronological_train_test_split(df_feat, train_ratio=0.8)

    max_train_date = train_df["timestamp"].max()
    min_test_date = test_df["timestamp"].min()

    assert max_train_date < min_test_date
    assert len(train_df) + len(test_df) == len(df_feat)


def test_preprocessor_fit_transform():
    """Verify that AirQualityPreprocessor transforms features and handles missing values."""
    df_dummy = pd.DataFrame({
        col: np.random.uniform(10, 100, size=30) for col in FEATURE_COLUMNS
    })
    # Introduce random NaNs
    df_dummy.iloc[2, 0] = np.nan
    df_dummy.iloc[5, 5] = np.nan

    preprocessor = AirQualityPreprocessor(FEATURE_COLUMNS)
    preprocessor.fit(df_dummy)
    assert preprocessor.is_fitted

    scaled = preprocessor.transform(df_dummy)
    assert scaled.shape == (30, len(FEATURE_COLUMNS))
    assert not np.isnan(scaled).any()

    # Single-sample inference test
    single_input = {
        "pm25_lag1": 85.0,
        "pm25_lag2": 90.0,
        "pm10_lag1": 150.0,
        "month": 11
    }
    single_scaled = preprocessor.transform_single_input(single_input)
    assert single_scaled.shape == (1, len(FEATURE_COLUMNS))
    assert not np.isnan(single_scaled).any()

