"""Data loader module for AirAware AI.
Loads, validates, and cleans historical air quality records.
"""

import os
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

# Standardized column name mapping
COLUMN_SYNONYMS = {
    "timestamp": ["date", "datetime", "timestamp", "time", "Date", "Timestamp", "DateTime"],
    "pm25": ["pm2.5", "pm25", "pm2_5", "pm2.5_atm", "PM2.5", "PM25", "pm_25"],
    "pm10": ["pm10", "pm_10", "PM10"],
    "no2": ["no2", "NO2", "nitrogen_dioxide"],
    "so2": ["so2", "SO2", "sulfur_dioxide"],
    "co": ["co", "CO", "carbon_monoxide"],
    "o3": ["o3", "O3", "ozone", "Ozone"]
}

# Physical bounds for pollutants (min, max) in µg/m³ (CO in mg/m³)
POLLUTANT_BOUNDS = {
    "pm25": (0.0, 1500.0),
    "pm10": (0.0, 2000.0),
    "no2": (0.0, 600.0),
    "so2": (0.0, 600.0),
    "co": (0.0, 100.0),
    "o3": (0.0, 600.0)
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map known synonyms of column names to lowercase standard names."""
    col_mapping = {}
    for col in df.columns:
        col_clean = str(col).strip()
        matched = False
        for standard_name, synonyms in COLUMN_SYNONYMS.items():
            if col_clean in synonyms or col_clean.lower() in [s.lower() for s in synonyms]:
                col_mapping[col] = standard_name
                matched = True
                break
        if not matched:
            col_mapping[col] = col_clean.lower().replace(" ", "_")
    return df.rename(columns=col_mapping)


def load_air_quality_data(filepath: str = "data/air_quality_data.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load, validate, and clean historical air quality dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Tuple of (Cleaned DataFrame, metadata dictionary)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            f"Please ensure 'data/air_quality_data.csv' exists or run the data setup."
        )

    raw_df = pd.read_csv(filepath)
    if raw_df.empty:
        raise ValueError(f"Dataset at '{filepath}' is empty.")

    raw_row_count = len(raw_df)

    # Standardize column headers
    df = standardize_columns(raw_df)

    # Verify required core columns
    if "timestamp" not in df.columns:
        raise ValueError(
            f"Dataset must contain a timestamp/date column. Found columns: {list(raw_df.columns)}"
        )
    if "pm25" not in df.columns:
        raise ValueError(
            f"Dataset must contain a PM2.5 target column. Found columns: {list(raw_df.columns)}"
        )

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Chronological sort
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Deduplicate timestamps, keeping the first observation
    df = df.drop_duplicates(subset=["timestamp"], keep="first")

    # Validate numeric pollutant values and enforce physical bounds
    for col, (min_val, max_val) in POLLUTANT_BOUNDS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Invalidate physical impossibilities (e.g. negative readings)
            df.loc[(df[col] < min_val) | (df[col] > max_val), col] = np.nan

    # Drop records where target PM2.5 is missing
    df_clean = df.dropna(subset=["pm25"]).copy().reset_index(drop=True)

    metadata = {
        "filepath": filepath,
        "raw_records": raw_row_count,
        "clean_records": len(df_clean),
        "start_date": str(df_clean["timestamp"].min()),
        "end_date": str(df_clean["timestamp"].max()),
        "columns_available": list(df_clean.columns),
        "target_column": "pm25",
        "units": {
            "pm25": "µg/m³",
            "pm10": "µg/m³",
            "no2": "µg/m³",
            "so2": "µg/m³",
            "co": "mg/m³",
            "o3": "µg/m³"
        }
    }

    return df_clean, metadata

