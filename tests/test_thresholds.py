"""Unit tests for PM2.5 air quality categorization and threshold logic."""

import pytest
import math
from src.air_quality import (
    categorize_pm25,
    get_threshold_config,
    get_category_order,
    CPCB_PM25_CATEGORIES,
    EPA_PM25_CATEGORIES
)


def test_cpcb_categories_standard_ranges():
    """Verify standard positive PM2.5 values map to the correct CPCB categories."""
    assert categorize_pm25(15.0, "CPCB")["category"] == "Good"
    assert categorize_pm25(45.0, "CPCB")["category"] == "Satisfactory"
    assert categorize_pm25(75.0, "CPCB")["category"] == "Moderate"
    assert categorize_pm25(105.0, "CPCB")["category"] == "Poor"
    assert categorize_pm25(180.0, "CPCB")["category"] == "Very Poor"
    assert categorize_pm25(320.0, "CPCB")["category"] == "Severe"


def test_cpcb_boundary_conditions():
    """Verify exact boundary values."""
    assert categorize_pm25(0.0, "CPCB")["category"] == "Good"
    assert categorize_pm25(30.0, "CPCB")["category"] == "Good"
    assert categorize_pm25(60.0, "CPCB")["category"] == "Satisfactory"
    assert categorize_pm25(90.0, "CPCB")["category"] == "Moderate"
    assert categorize_pm25(120.0, "CPCB")["category"] == "Poor"
    assert categorize_pm25(250.0, "CPCB")["category"] == "Very Poor"
    assert categorize_pm25(250.1, "CPCB")["category"] == "Severe"


def test_epa_categories_standard_ranges():
    """Verify EPA standard category mapping."""
    assert categorize_pm25(8.0, "EPA")["category"] == "Good"
    assert categorize_pm25(20.0, "EPA")["category"] == "Moderate"
    assert categorize_pm25(40.0, "EPA")["category"] == "Unhealthy for Sensitive Groups"
    assert categorize_pm25(80.0, "EPA")["category"] == "Unhealthy"
    assert categorize_pm25(180.0, "EPA")["category"] == "Very Unhealthy"
    assert categorize_pm25(350.0, "EPA")["category"] == "Hazardous"


def test_negative_values():
    """Negative PM2.5 readings are physically invalid and must be rejected."""
    res = categorize_pm25(-10.5, "CPCB")
    assert not res["is_valid"]
    assert "Invalid" in res["category"]


def test_nan_and_none_values():
    """Missing and NaN values must be safely handled without throwing uncaught exceptions."""
    res_none = categorize_pm25(None)
    assert not res_none["is_valid"]
    assert res_none["category"] == "Unknown"

    res_nan = categorize_pm25(float("nan"))
    assert not res_nan["is_valid"]
    assert res_nan["category"] == "Unknown"


def test_extreme_high_values():
    """Verify extreme peak air pollution values still map to Severe/Hazardous."""
    res = categorize_pm25(999.0, "CPCB")
    assert res["category"] == "Severe"
    assert res["is_valid"]

