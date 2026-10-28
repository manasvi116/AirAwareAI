"""Air quality categorization and threshold definitions.

Aligned with:
1. Central Pollution Control Board (CPCB) National Air Quality Index (NAQI), India.
2. United States Environmental Protection Agency (US EPA) Air Quality Index.

Disclaimer:
Categories derived here are based specifically on PM2.5 (Particulate Matter <= 2.5 µm)
concentration and serve as an educational/informative estimate. An official multi-pollutant
AQI accounts for multiple criteria pollutants (PM10, NO2, SO2, CO, O3, NH3, Pb) using a sub-index max rule.
"""

from typing import Dict, Any, List, Optional
import math

# CPCB PM2.5 Concentration Breakpoints (24-hour average in µg/m³)
# Ref: Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Govt of India
CPCB_PM25_CATEGORIES = {
    "Good": {
        "min": 0.0,
        "max": 30.0,
        "color": "#16a34a",       # Emerald green
        "bg_light": "#dcfce7",
        "description": "Minimal health impact.",
        "health_advisory": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
        "caution": "Great day to be outdoors and enjoy physical activities."
    },
    "Satisfactory": {
        "min": 30.0,
        "max": 60.0,
        "color": "#65a30d",       # Lime green
        "bg_light": "#ecfccb",
        "description": "Minor breathing discomfort to sensitive people.",
        "health_advisory": "Sensitive individuals (asthma, children, elderly) may experience mild respiratory symptoms.",
        "caution": "Unusually sensitive people should consider reducing prolonged outdoor exertion."
    },
    "Moderate": {
        "min": 60.0,
        "max": 90.0,
        "color": "#ca8a04",       # Yellow-amber
        "bg_light": "#fef9c3",
        "description": "Breathing discomfort to the people with lungs, asthma and heart diseases.",
        "health_advisory": "Acceptable air quality for general public, but potential discomfort for vulnerable groups.",
        "caution": "People with respiratory or cardiac conditions should limit heavy outdoor exertion."
    },
    "Poor": {
        "min": 90.0,
        "max": 120.0,
        "color": "#ea580c",       # Orange
        "bg_light": "#ffedd5",
        "description": "Breathing discomfort to most people on prolonged exposure.",
        "health_advisory": "Everyone may begin to experience health effects; members of sensitive groups may experience more serious effects.",
        "caution": "Avoid strenuous outdoor exertion; consider wearing an N95 mask if outdoors for long periods."
    },
    "Very Poor": {
        "min": 120.0,
        "max": 250.0,
        "color": "#dc2626",       # Red
        "bg_light": "#fee2e2",
        "description": "Respiratory illness on prolonged exposure.",
        "health_advisory": "Health alert: The risk of health effects is increased for everyone.",
        "caution": "Avoid outdoor physical activities; keep windows closed; run air purifiers indoors where possible."
    },
    "Severe": {
        "min": 250.0,
        "max": float("inf"),
        "color": "#7f1d1d",       # Dark Maroon
        "bg_light": "#fef2f2",
        "description": "Affects healthy people and seriously impacts those with existing diseases.",
        "health_advisory": "Health warning of emergency conditions: The entire population is likely to be affected.",
        "caution": "Stay indoors; avoid all physical outdoor exertion; use air purifiers and sealed filtration."
    }
}

# EPA PM2.5 Breakpoints (µg/m³, 24-hr average standard)
EPA_PM25_CATEGORIES = {
    "Good": {
        "min": 0.0,
        "max": 12.0,
        "color": "#16a34a",
        "bg_light": "#dcfce7",
        "description": "Air quality is satisfactory, poses little or no risk.",
        "health_advisory": "None",
        "caution": "Ideal for outdoor activities."
    },
    "Moderate": {
        "min": 12.0,
        "max": 35.4,
        "color": "#ca8a04",
        "bg_light": "#fef9c3",
        "description": "Acceptable; modest risk for unusually sensitive people.",
        "health_advisory": "Sensitive people should consider reducing prolonged or heavy outdoor exertion.",
        "caution": "Unusually sensitive people should consider reducing prolonged exertion."
    },
    "Unhealthy for Sensitive Groups": {
        "min": 35.4,
        "max": 55.4,
        "color": "#ea580c",
        "bg_light": "#ffedd5",
        "description": "Members of sensitive groups may experience health effects.",
        "health_advisory": "People with heart/lung disease, older adults, and children should reduce prolonged exertion.",
        "caution": "Sensitive groups take breaks and do less intense outdoor activities."
    },
    "Unhealthy": {
        "min": 55.4,
        "max": 150.4,
        "color": "#dc2626",
        "bg_light": "#fee2e2",
        "description": "Everyone may begin to experience health effects.",
        "health_advisory": "Health alert: Increased risk for general public.",
        "caution": "Avoid prolonged outdoor exertion; wear appropriate masks."
    },
    "Very Unhealthy": {
        "min": 150.4,
        "max": 250.4,
        "color": "#9333ea",
        "bg_light": "#f3e8ff",
        "description": "Health alert: serious risk for the entire population.",
        "health_advisory": "Significant respiratory and cardiovascular aggravation.",
        "caution": "Avoid any strenuous outdoor activities."
    },
    "Hazardous": {
        "min": 250.4,
        "max": float("inf"),
        "color": "#7f1d1d",
        "bg_light": "#fef2f2",
        "description": "Health warning of emergency conditions.",
        "health_advisory": "Emergency warning: serious effects on entire population.",
        "caution": "Remain indoors with clean air systems; avoid outdoor exposure."
    }
}

CPCB_ORDER: List[str] = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
EPA_ORDER: List[str] = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]


def get_threshold_config(standard: str = "CPCB") -> Dict[str, Dict[str, Any]]:
    """Returns the threshold configuration for the specified air quality standard."""
    if standard.upper() == "EPA":
        return EPA_PM25_CATEGORIES
    return CPCB_PM25_CATEGORIES


def get_category_order(standard: str = "CPCB") -> List[str]:
    """Returns the ordered list of category labels from cleanest to most polluted."""
    if standard.upper() == "EPA":
        return EPA_ORDER if standard.upper() == "EPA" else CPCB_ORDER
    return CPCB_ORDER


def categorize_pm25(value: Optional[float], standard: str = "CPCB") -> Dict[str, Any]:
    """
    Categorize a PM2.5 concentration value into an air quality band.

    Args:
        value: PM2.5 concentration in µg/m³ (numerical).
        standard: 'CPCB' (default, India NAQI) or 'EPA' (US EPA).

    Returns:
        Dictionary containing category name, color codes, health advisory, and descriptions.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return {
            "category": "Unknown",
            "color": "#64748b",
            "bg_light": "#f1f5f9",
            "description": "Invalid or missing PM2.5 reading.",
            "health_advisory": "Unable to determine category from missing or non-numerical input.",
            "caution": "Ensure valid numerical pollutant inputs are provided.",
            "standard_used": standard,
            "is_valid": False
        }

    try:
        val = float(value)
    except (ValueError, TypeError):
        return {
            "category": "Unknown",
            "color": "#64748b",
            "bg_light": "#f1f5f9",
            "description": "Non-numerical input encountered.",
            "health_advisory": "Invalid measurement input.",
            "caution": "Please enter a valid numeric value.",
            "standard_used": standard,
            "is_valid": False
        }

    if val < 0:
        return {
            "category": "Invalid (Negative)",
            "color": "#ef4444",
            "bg_light": "#fee2e2",
            "description": f"Physical error: PM2.5 concentration cannot be negative ({val:.1f} µg/m³).",
            "health_advisory": "Sensor malfunction or erroneous input.",
            "caution": "Verify sensor calibration or input format.",
            "standard_used": standard,
            "is_valid": False
        }

    config = get_threshold_config(standard)

    for cat_name, info in config.items():
        min_val = info["min"]
        max_val = info["max"]
        # Handle lower bound inclusive, upper bound exclusive (except for infinite highest bracket)
        if min_val <= val <= max_val or (val > min_val and max_val == float("inf")):
            return {
                "category": cat_name,
                "color": info["color"],
                "bg_light": info["bg_light"],
                "description": info["description"],
                "health_advisory": info["health_advisory"],
                "caution": info["caution"],
                "pm25_value": val,
                "standard_used": standard,
                "is_valid": True
            }

    # Fallback to the most severe category if above all upper bounds
    last_cat = list(config.keys())[-1]
    info = config[last_cat]
    return {
        "category": last_cat,
        "color": info["color"],
        "bg_light": info["bg_light"],
        "description": info["description"],
        "health_advisory": info["health_advisory"],
        "caution": info["caution"],
        "pm25_value": val,
        "standard_used": standard,
        "is_valid": True
    }

