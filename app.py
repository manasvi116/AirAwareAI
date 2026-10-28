"""AirAware AI — Air Quality Prediction System
Streamlit Web Application.

A college project for AICTE–BharatCares–IBM SkillsBuild Machine Learning and Applied AI Internship.
Aligned with UN SDG 3 (Good Health and Well-Being) and SDG 11 (Sustainable Cities and Communities).
"""

import os
import sys

# Ensure root directory is on Python search path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.data_loader import load_air_quality_data
from src.air_quality import (
    categorize_pm25,
    get_threshold_config,
    get_category_order,
    CPCB_PM25_CATEGORIES,
    EPA_PM25_CATEGORIES
)
from src.preprocessing import FEATURE_COLUMNS, AirQualityPreprocessor


# Page Configuration
st.set_page_config(
    page_title="AirAware AI — Air Quality Prediction System",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        border-radius: 16px;
        padding: 32px 36px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
        color: #f8fafc;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        max-width: 850px;
        line-height: 1.5;
        margin-bottom: 18px;
    }
    .hero-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .hero-badge-active {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
    }

    /* Metric Cards */
    .metric-grid-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    .metric-grid-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.07);
    }
    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }
    .metric-icon {
        font-size: 1.25rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        margin: 4px 0 6px 0;
    }
    .metric-subtitle {
        font-size: 0.82rem;
        color: #94a3b8;
    }

    /* Section Headers */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e293b;
        margin: 28px 0 12px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Result Card */
    .result-card {
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08);
        border-width: 2px;
        border-style: solid;
    }
    .result-badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 9999px;
        color: #ffffff;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.02em;
        margin-top: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.12);
    }

    /* Health Advice Cards */
    .health-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    .health-card-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #1e293b;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .health-card-desc {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.45;
    }

    /* Preset Chips */
    .preset-container {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 18px;
    }

    /* Alert / Disclaimer */
    .disclaimer-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #92400e;
        margin-top: 20px;
    }

    /* SDG Card */
    .sdg-box {
        border-radius: 14px;
        padding: 24px 26px;
        height: 100%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        border-width: 2px;
        border-style: solid;
    }
</style>
""", unsafe_allow_html=True)


# Cache loading of data and models
@st.cache_data
def get_clean_data():
    data_path = os.path.join("data", "air_quality_data.csv")
    if not os.path.exists(data_path):
        return None, None
    try:
        return load_air_quality_data(data_path)
    except Exception as e:
        return None, str(e)


@st.cache_resource
def get_model_artifacts():
    summary_path = os.path.join("models", "training_summary.joblib")
    preproc_path = os.path.join("models", "preprocessor.joblib")
    best_reg_path = os.path.join("models", "best_regressor.joblib")
    all_reg_path = os.path.join("models", "all_regressors.joblib")
    best_clf_path = os.path.join("models", "best_classifier.joblib")
    all_clf_path = os.path.join("models", "all_classifiers.joblib")

    if not all(os.path.exists(p) for p in [summary_path, preproc_path, best_reg_path]):
        return None

    try:
        summary = joblib.load(summary_path)
        preprocessor = joblib.load(preproc_path)
        best_reg = joblib.load(best_reg_path)
        all_reg = joblib.load(all_reg_path) if os.path.exists(all_reg_path) else None
        best_clf = joblib.load(best_clf_path) if os.path.exists(best_clf_path) else None
        all_clf = joblib.load(all_clf_path) if os.path.exists(all_clf_path) else None
        return {
            "summary": summary,
            "preprocessor": preprocessor,
            "best_regressor": best_reg,
            "all_regressors": all_reg,
            "best_classifier": best_clf,
            "all_classifiers": all_clf
        }
    except Exception as e:
        return None


def create_pm25_gauge(value: float, category_info: dict, max_scale: float = 350.0):
    """Generate a clean Plotly radial gauge / speedometer chart for PM2.5."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "PM2.5 (µg/m³)", 'font': {'size': 18, 'color': '#1e293b', 'family': 'Plus Jakarta Sans'}},
        number={'suffix': " µg/m³", 'font': {'size': 26, 'color': category_info["color"], 'family': 'Plus Jakarta Sans', 'weight': 'bold'}},
        gauge={
            'axis': {'range': [0, max_scale], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': category_info["color"], 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 30], 'color': "rgba(22, 163, 74, 0.25)"},
                {'range': [30, 60], 'color': "rgba(101, 163, 13, 0.25)"},
                {'range': [60, 90], 'color': "rgba(202, 138, 4, 0.25)"},
                {'range': [90, 120], 'color': "rgba(234, 88, 12, 0.25)"},
                {'range': [120, 250], 'color': "rgba(220, 38, 38, 0.25)"},
                {'range': [250, max_scale], 'color': "rgba(127, 29, 29, 0.35)"}
            ],
            'threshold': {
                'line': {'color': category_info["color"], 'width': 4},
                'thickness': 0.85,
                'value': value
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=35, b=20),
        font={'family': "Plus Jakarta Sans"}
    )
    return fig


# Sidebar Navigation & Branding
st.sidebar.markdown("""
<div style="padding: 10px 0 16px 0;">
    <div style="font-size:1.45rem; font-weight:800; color:#0f172a; display:flex; align-items:center; gap:8px;">
        🍃 AirAware AI
    </div>
    <div style="font-size:0.82rem; font-weight:600; color:#059669; text-transform:uppercase; letter-spacing:0.04em;">
        Machine Learning System
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    "**AICTE–BharatCares–IBM SkillsBuild**\n\n"
    "*Applied AI Internship Project*"
)
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Dashboard Overview",
        "🔮 Real-Time PM2.5 Predictor",
        "📈 Model Evaluation & Benchmarks",
        "📁 Dataset Explorer",
        "🎯 UN SDG Alignment"
    ],
    index=0
)

st.sidebar.markdown("---")
standard_selection = st.sidebar.selectbox(
    "Regulatory Standard",
    ["CPCB (National AQI, India)", "US EPA (United States)"],
    index=0,
    help="Select the air quality standard thresholds for category mapping."
)
standard_code = "CPCB" if "CPCB" in standard_selection else "EPA"

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background-color:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px; font-size:0.75rem; color:#64748b; line-height:1.4;">
    <strong>Academic Prototype:</strong> Developed for educational demonstration under the IBM SkillsBuild applied AI internship. Predictions are statistical estimates.
</div>
""", unsafe_allow_html=True)


# Load Data and Model Artifacts
df_clean, meta_or_err = get_clean_data()
artifacts = get_model_artifacts()


# -----------------------------------------------------------------------------
# TAB 1: DASHBOARD OVERVIEW
# -----------------------------------------------------------------------------
if nav_choice == "📊 Dashboard Overview":
    # Hero Banner
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Air Quality Intelligence & Forecasting</div>
        <div class="hero-subtitle">
            An applied machine learning system predicting fine particulate matter (PM2.5) concentrations 
            and classifying air quality bands using authentic Central Pollution Control Board (CPCB) monitoring data.
        </div>
        <div class="hero-badge-row">
            <span class="hero-badge hero-badge-active">● Models Trained & Operational</span>
            <span class="hero-badge">AICTE–BharatCares–IBM SkillsBuild</span>
            <span class="hero-badge">UN SDG 3: Good Health</span>
            <span class="hero-badge">UN SDG 11: Sustainable Cities</span>
            <span class="hero-badge">Delhi Urban Network (2015–2020)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df_clean is None or artifacts is None:
        st.warning("Data or trained models not found. Please train models by running `python train_models.py` in your terminal.")
        if st.button("🚀 Train Models Now"):
            with st.spinner("Training models in background..."):
                from train_models import run_training
                run_training()
                st.success("Models trained successfully! Please reload the page.")
        st.stop()

    summary = artifacts["summary"]
    reg_summary = summary["regression"]
    split = summary["split_info"]

    # Metric Scorecards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-grid-card">
            <div class="metric-header">
                <span class="metric-label">Dataset Records</span>
                <span class="metric-icon">📑</span>
            </div>
            <div class="metric-value">{len(df_clean):,}</div>
            <div class="metric-subtitle">Validated daily observations</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        mean_pm25 = df_clean["pm25"].mean()
        st.markdown(f"""
        <div class="metric-grid-card">
            <div class="metric-header">
                <span class="metric-label">Historical Mean PM2.5</span>
                <span class="metric-icon">🌪️</span>
            </div>
            <div class="metric-value">{mean_pm25:.1f} <span style="font-size:1rem; font-weight:500;">µg/m³</span></div>
            <div class="metric-subtitle">Ambient baseline average</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        best_r2 = reg_summary["metrics"][reg_summary["best_model_name"]]["test"]["R2"]
        st.markdown(f"""
        <div class="metric-grid-card">
            <div class="metric-header">
                <span class="metric-label">Best Regressor R²</span>
                <span class="metric-icon">🎯</span>
            </div>
            <div class="metric-value">{best_r2:.3f}</div>
            <div class="metric-subtitle">{reg_summary['best_model_name']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        best_mae = reg_summary["metrics"][reg_summary["best_model_name"]]["test"]["MAE"]
        st.markdown(f"""
        <div class="metric-grid-card">
            <div class="metric-header">
                <span class="metric-label">Held-out Test MAE</span>
                <span class="metric-icon">📉</span>
            </div>
            <div class="metric-value">{best_mae:.1f} <span style="font-size:1rem; font-weight:500;">µg/m³</span></div>
            <div class="metric-subtitle">On 400 test days (2019–2020)</div>
        </div>
        """, unsafe_allow_html=True)

    # Historical PM2.5 Time Series Trend with Range Selector
    st.markdown('<div class="section-title">📈 Longitudinal PM2.5 Trend (2015 – 2020)</div>', unsafe_allow_html=True)
    st.markdown("Explore ambient particulate dynamics across 5.5 years with regulatory threshold reference bands:")

    fig_trend = go.Figure()

    # 30-day Rolling Mean for trend clarity
    df_clean["pm25_30d_smooth"] = df_clean["pm25"].rolling(window=30, min_periods=7).mean()

    # Base daily series
    fig_trend.add_trace(go.Scatter(
        x=df_clean["timestamp"],
        y=df_clean["pm25"],
        mode="lines",
        name="Daily PM2.5",
        line=dict(color="rgba(37, 99, 235, 0.4)", width=1.2),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Daily PM2.5:</b> %{y:.1f} µg/m³<extra></extra>"
    ))

    # Smoothed trendline
    fig_trend.add_trace(go.Scatter(
        x=df_clean["timestamp"],
        y=df_clean["pm25_30d_smooth"],
        mode="lines",
        name="30-Day Moving Average",
        line=dict(color="#0f766e", width=2.4),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>30-Day Avg:</b> %{y:.1f} µg/m³<extra></extra>"
    ))

    # CPCB Threshold lines
    fig_trend.add_hline(y=30, line_dash="dot", line_color="#16a34a", annotation_text="CPCB Good (30)", annotation_position="top left")
    fig_trend.add_hline(y=60, line_dash="dot", line_color="#65a30d", annotation_text="CPCB Satisfactory (60)", annotation_position="top left")
    fig_trend.add_hline(y=90, line_dash="dot", line_color="#ca8a04", annotation_text="CPCB Moderate (90)", annotation_position="top left")
    fig_trend.add_hline(y=120, line_dash="dot", line_color="#ea580c", annotation_text="CPCB Poor (120)", annotation_position="top left")
    fig_trend.add_hline(y=250, line_dash="dot", line_color="#dc2626", annotation_text="CPCB Very Poor (250)", annotation_position="top left")

    fig_trend.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All 5.5 Years")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis_title="PM2.5 Concentration (µg/m³)",
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=20, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Category Distribution & Seasonality Analysis
    st.markdown('<div class="section-title">🏷️ Air Quality Distribution & Seasonal Profile</div>', unsafe_allow_html=True)
    col_chart1, col_chart2 = st.columns([1, 1.2])

    categories_list = [categorize_pm25(v, standard=standard_code)["category"] for v in df_clean["pm25"]]
    cat_counts = pd.Series(categories_list).value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]

    cfg = get_threshold_config(standard_code)
    cat_order = get_category_order(standard_code)
    cat_counts["Order"] = cat_counts["Category"].apply(lambda c: cat_order.index(c) if c in cat_order else 99)
    cat_counts = cat_counts.sort_values("Order")
    color_map = {k: v["color"] for k, v in cfg.items()}

    with col_chart1:
        fig_donut = px.pie(
            cat_counts,
            names="Category",
            values="Count",
            hole=0.52,
            color="Category",
            color_discrete_map=color_map,
            title=f"Category Distribution ({standard_code} Standards)"
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent+label", hoverinfo="label+value+percent")
        fig_donut.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), font={'family': 'Plus Jakarta Sans'})
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_chart2:
        # Seasonal Monthly Pollution Profile
        df_clean["month_num"] = df_clean["timestamp"].dt.month
        month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
        df_clean["month_name"] = df_clean["month_num"].map(month_names)

        monthly_avg = df_clean.groupby(["month_num", "month_name"])["pm25"].mean().reset_index()
        monthly_avg = monthly_avg.sort_values("month_num")

        # Color bars by CPCB tier
        monthly_avg["color"] = monthly_avg["pm25"].apply(lambda v: categorize_pm25(v, standard="CPCB")["color"])

        fig_season = go.Figure(go.Bar(
            x=monthly_avg["month_name"],
            y=monthly_avg["pm25"],
            marker_color=monthly_avg["color"],
            text=monthly_avg["pm25"].round(1),
            textposition="auto",
            hovertemplate="<b>Month:</b> %{x}<br><b>Mean PM2.5:</b> %{y:.1f} µg/m³<extra></extra>"
        ))
        fig_season.update_layout(
            title="Monthly Seasonal Dynamics (Winter Peak vs Monsoon Washout)",
            xaxis_title="Month of Year",
            yaxis_title="Average PM2.5 (µg/m³)",
            template="plotly_white",
            height=380,
            margin=dict(l=30, r=30, t=40, b=20),
            font={'family': 'Plus Jakarta Sans'}
        )
        st.plotly_chart(fig_season, use_container_width=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Scientific & Ethical Protocol:</strong> AirAware AI employs a strictly backwards-looking, chronological feature engineering pipeline.
        Category tiers are derived directly from the numerical PM2.5 concentration and serve as an educational prototype. 
        Statutory multi-pollutant AQI declarations must always be verified with the official CPCB portal.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 2: PREDICTION INTERFACE
# -----------------------------------------------------------------------------
elif nav_choice == "🔮 Real-Time PM2.5 Predictor":
    st.markdown("""
    <div class="hero-container" style="padding: 24px 30px; margin-bottom: 20px;">
        <div style="font-size: 1.85rem; font-weight: 800; color: #ffffff;">🔮 PM2.5 Concentration & Air Quality Predictor</div>
        <div style="font-size: 0.95rem; color: #cbd5e1; margin-top: 4px;">
            Input pollutant observations up to day <i>t</i> to generate a machine learning forecast for day <i>t+1</i> with actionable health guidelines.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if artifacts is None:
        st.error("Model artifacts not found. Please train models first using `python train_models.py`.")
        st.stop()

    preprocessor = artifacts["preprocessor"]
    best_regressor = artifacts["best_regressor"]
    all_regressors = artifacts["all_regressors"]
    reg_summary = artifacts["summary"]["regression"]

    # Preset selection card
    st.markdown('<div class="section-title">⚡ 1-Click Scenario Profiles</div>', unsafe_allow_html=True)
    st.markdown("Quickly test realistic atmospheric scenarios representative of Delhi's seasons:")

    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

    # Session state for preset defaults
    if "selected_preset" not in st.session_state:
        st.session_state.selected_preset = "🏙️ Moderate Urban Day"

    with preset_col1:
        if st.button("🌿 Clean Monsoon Day", use_container_width=True):
            st.session_state.selected_preset = "🌿 Clean Monsoon Day"
    with preset_col2:
        if st.button("🏙️ Moderate Urban Day", use_container_width=True):
            st.session_state.selected_preset = "🏙️ Moderate Urban Day"
    with preset_col3:
        if st.button("🌫️ Winter Smog Episode", use_container_width=True):
            st.session_state.selected_preset = "🌫️ Winter Smog Episode"
    with preset_col4:
        if st.button("⚙️ Custom Configuration", use_container_width=True):
            st.session_state.selected_preset = "⚙️ Custom Configuration"

    st.info(f"Active Scenario: **{st.session_state.selected_preset}**")

    # Load defaults based on scenario
    if st.session_state.selected_preset == "🌿 Clean Monsoon Day":
        default_pm25_1, default_pm25_2, default_pm25_3, default_pm25_7, default_roll7 = 24.0, 28.0, 32.0, 26.0, 27.5
        default_pm10, default_no2, default_so2, default_co, default_o3 = 45.0, 18.0, 7.5, 0.6, 22.0
        default_month = 8  # August (Monsoon)
    elif st.session_state.selected_preset == "🏙️ Moderate Urban Day":
        default_pm25_1, default_pm25_2, default_pm25_3, default_pm25_7, default_roll7 = 72.0, 68.0, 80.0, 65.0, 71.0
        default_pm10, default_no2, default_so2, default_co, default_o3 = 140.0, 42.0, 14.0, 1.2, 35.0
        default_month = 4  # April (Spring/Summer)
    elif st.session_state.selected_preset == "🌫️ Winter Smog Episode":
        default_pm25_1, default_pm25_2, default_pm25_3, default_pm25_7, default_roll7 = 265.0, 240.0, 285.0, 210.0, 250.0
        default_pm10, default_no2, default_so2, default_co, default_o3 = 420.0, 95.0, 24.0, 2.8, 48.0
        default_month = 11  # November (Winter Inversion)
    else:
        default_pm25_1, default_pm25_2, default_pm25_3, default_pm25_7, default_roll7 = 95.0, 92.0, 100.0, 88.0, 93.0
        default_pm10, default_no2, default_so2, default_co, default_o3 = 185.0, 45.0, 12.0, 1.1, 38.0
        default_month = 10

    # Model selector dropdown
    selected_model_name = st.selectbox(
        "Select Regression Model Engine",
        list(all_regressors.keys()) if all_regressors else [reg_summary["best_model_name"]],
        index=0,
        help="Linear Regression demonstrated the lowest out-of-sample RMSE on held-out test data."
    )
    active_model = all_regressors[selected_model_name] if all_regressors else best_regressor

    with st.form("interactive_prediction_form"):
        st.markdown('<div class="section-title">1. Time-Lagged PM2.5 Series (µg/m³)</div>', unsafe_allow_html=True)
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            pm25_lag1 = st.number_input("Current Day PM2.5 (Day t)", min_value=0.0, max_value=1500.0, value=float(default_pm25_1), step=1.0)
            pm25_lag2 = st.number_input("Yesterday PM2.5 (Day t-1)", min_value=0.0, max_value=1500.0, value=float(default_pm25_2), step=1.0)
        with col_p2:
            pm25_lag3 = st.number_input("Day Before Yesterday (Day t-2)", min_value=0.0, max_value=1500.0, value=float(default_pm25_3), step=1.0)
            pm25_lag7 = st.number_input("Same Day Last Week (Day t-6)", min_value=0.0, max_value=1500.0, value=float(default_pm25_7), step=1.0)
        with col_p3:
            pm25_roll7 = st.number_input("7-Day Rolling Average PM2.5", min_value=0.0, max_value=1500.0, value=float(default_roll7), step=1.0)
            target_month = st.slider("Target Forecast Month (t+1)", min_value=1, max_value=12, value=int(default_month))

        st.markdown('<div class="section-title">2. Ambient Co-Pollutant Concentrations at Day t</div>', unsafe_allow_html=True)
        col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
        with col_c1:
            pm10_lag1 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=2000.0, value=float(default_pm10), step=1.0)
        with col_c2:
            no2_lag1 = st.number_input("NO2 (µg/m³)", min_value=0.0, max_value=600.0, value=float(default_no2), step=1.0)
        with col_c3:
            so2_lag1 = st.number_input("SO2 (µg/m³)", min_value=0.0, max_value=600.0, value=float(default_so2), step=0.5)
        with col_c4:
            co_lag1 = st.number_input("CO (mg/m³)", min_value=0.0, max_value=100.0, value=float(default_co), step=0.1)
        with col_c5:
            o3_lag1 = st.number_input("Ozone O3 (µg/m³)", min_value=0.0, max_value=600.0, value=float(default_o3), step=1.0)

        predict_btn = st.form_submit_button("⚡ Compute Next-Day PM2.5 Forecast", use_container_width=True)

    if predict_btn:
        # Validate inputs
        if any(v < 0 for v in [pm25_lag1, pm25_lag2, pm25_lag3, pm25_lag7, pm25_roll7, pm10_lag1, no2_lag1, so2_lag1, co_lag1, o3_lag1]):
            st.error("Validation error: Atmospheric pollutant measurements cannot be negative.")
        else:
            # Construct input sample dictionary matching preprocessor schema
            input_dict = {
                "pm25_lag1": pm25_lag1,
                "pm25_lag2": pm25_lag2,
                "pm25_lag3": pm25_lag3,
                "pm25_lag7": pm25_lag7,
                "pm25_roll_mean_7": pm25_roll7,
                "pm10_lag1": pm10_lag1,
                "no2_lag1": no2_lag1,
                "so2_lag1": so2_lag1,
                "co_lag1": co_lag1,
                "o3_lag1": o3_lag1,
                "month": target_month,
                "day_of_week": 2,          # Representative mid-week Wednesday
                "day_of_year": target_month * 30,
                "is_weekend": 0,
                "sin_month": np.sin(2 * np.pi * target_month / 12.0),
                "cos_month": np.cos(2 * np.pi * target_month / 12.0)
            }

            # Preprocess using exact fitted training transformation
            X_input_scaled = preprocessor.transform_single_input(input_dict)

            # Predict PM2.5 numerical concentration
            predicted_pm25 = float(active_model.predict(X_input_scaled)[0])
            predicted_pm25 = max(0.0, predicted_pm25)

            # Categorize prediction
            cat_info = categorize_pm25(predicted_pm25, standard=standard_code)

            st.markdown("---")
            st.markdown('<div class="section-title">🎯 Model Prediction Summary (Day t+1 Horizon)</div>', unsafe_allow_html=True)

            col_res_dial, col_res_card = st.columns([1.1, 1])

            with col_res_dial:
                fig_gauge = create_pm25_gauge(predicted_pm25, cat_info)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_res_card:
                delta_val = predicted_pm25 - pm25_lag1
                trend_symbol = "🔺 Increasing" if delta_val > 1 else "🔻 Improving" if delta_val < -1 else "➡️ Steady"
                trend_color = "#dc2626" if delta_val > 1 else "#16a34a" if delta_val < -1 else "#64748b"

                st.markdown(f"""
                <div class="result-card" style="background-color:{cat_info['bg_light']}; border-color:{cat_info['color']};">
                    <div style="font-size:0.85rem; font-weight:700; color:{cat_info['color']}; text-transform:uppercase; letter-spacing:0.06em;">
                        Predicted Air Quality Classification
                    </div>
                    <div style="font-size:3.1rem; font-weight:800; color:{cat_info['color']}; margin: 4px 0;">
                        {predicted_pm25:.1f} <span style="font-size:1.25rem; font-weight:500;">µg/m³</span>
                    </div>
                    <div class="result-badge" style="background-color:{cat_info['color']};">
                        {cat_info['category']}
                    </div>
                    <div style="margin-top:14px; font-size:0.92rem; font-weight:600; color:{trend_color};">
                        {trend_symbol}: {delta_val:+.1f} µg/m³ compared to Day t
                    </div>
                    <div style="font-size:0.8rem; color:#64748b; margin-top:6px;">
                        Engine: <code>{selected_model_name}</code> · Standard: <code>{standard_code}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 4 Actionable Health Recommendation Cards
            st.markdown('<div class="section-title">🛡️ Public Health & Precautionary Directives</div>', unsafe_allow_html=True)

            h_col1, h_col2, h_col3, h_col4 = st.columns(4)

            # Derive actionable guidance
            if cat_info['category'] in ["Good", "Satisfactory"]:
                act_outdoor = "Ideal for morning runs, cycling, and outdoor recreation."
                act_vent = "Keep windows open for natural ventilation and fresh airflow."
                act_mask = "No masks necessary for healthy individuals."
                act_sens = "Safe for elderly, children, and individuals with respiratory conditions."
            elif cat_info['category'] == "Moderate":
                act_outdoor = "Normal activities permitted; reduce intense prolonged exertion."
                act_vent = "Standard indoor ventilation; monitor sensitive family members."
                act_mask = "Optional N95 masks for individuals with chronic asthma or COPD."
                act_sens = "Sensitive individuals should take more breaks during outdoor activities."
            elif cat_info['category'] == "Poor":
                act_outdoor = "Avoid prolonged or strenuous outdoor physical exertion."
                act_vent = "Close windows during peak traffic hours; maintain clean indoor air."
                act_mask = "Wear an N95/FFP2 respirator when commuting or outdoors."
                act_sens = "Children and elderly should minimize outdoor exposure."
            else: # Very Poor or Severe
                act_outdoor = "Avoid all outdoor physical exertion; stay strictly indoors."
                act_vent = "Keep all windows sealed; activate HEPA air purifiers indoors."
                act_mask = "Mandatory certified N95/N99 respirators if venturing outside."
                act_sens = "Medical alert for cardiac and asthmatic patients; seek care if breathless."

            with h_col1:
                st.markdown(f"""
                <div class="health-card">
                    <div class="health-card-title">🏃 Outdoor Exercise</div>
                    <div class="health-card-desc">{act_outdoor}</div>
                </div>
                """, unsafe_allow_html=True)

            with h_col2:
                st.markdown(f"""
                <div class="health-card">
                    <div class="health-card-title">🪟 Indoor Ventilation</div>
                    <div class="health-card-desc">{act_vent}</div>
                </div>
                """, unsafe_allow_html=True)

            with h_col3:
                st.markdown(f"""
                <div class="health-card">
                    <div class="health-card-title">😷 Protective Masks</div>
                    <div class="health-card-desc">{act_mask}</div>
                </div>
                """, unsafe_allow_html=True)

            with h_col4:
                st.markdown(f"""
                <div class="health-card">
                    <div class="health-card-title">👶 Sensitive Demographics</div>
                    <div class="health-card-desc">{act_sens}</div>
                </div>
                """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 3: MODEL EVALUATION
# -----------------------------------------------------------------------------
elif nav_choice == "📈 Model Evaluation & Benchmarks":
    st.markdown("""
    <div class="hero-container" style="padding: 24px 30px; margin-bottom: 20px;">
        <div style="font-size: 1.85rem; font-weight: 800; color: #ffffff;">📈 Model Performance & Comparative Evaluation</div>
        <div style="font-size: 0.95rem; color: #cbd5e1; margin-top: 4px;">
            Scientific evaluation on held-out chronological test data (2019–2020) with strict data leakage prevention.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if artifacts is None:
        st.error("Model artifacts not found. Please train models first using `python train_models.py`.")
        st.stop()

    summary = artifacts["summary"]
    reg = summary["regression"]
    clf = summary["classification"]
    split = summary["split_info"]

    # Methodology Card
    st.markdown(f"""
    <div style="background-color:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; box-shadow:0 2px 4px rgba(0,0,0,0.02); margin-bottom:24px;">
        <div style="font-weight:700; font-size:1.05rem; color:#0f172a; margin-bottom:8px;">
            🛡️ Strict Chronological Partitioning (Leakage-Free)
        </div>
        <div style="font-size:0.9rem; color:#475569; line-height:1.6;">
            • <strong>Chronological Split:</strong> Shuffling is prohibited in time-series forecasting. The earliest 80% (<code>{split['train_start_date'][:10]}</code> to <code>{split['train_end_date'][:10]}</code>, {split['train_records']} records) forms the training set; the subsequent 20% (<code>{split['test_start_date'][:10]}</code> to <code>{split['test_end_date'][:10]}</code>, {split['test_records']} records) is held out.<br>
            • <strong>Preconditioning Isolation:</strong> Median imputation and standard scaling are fitted <em>strictly on the training split</em> and applied downstream.<br>
            • <strong>Backwards-Looking Feature Lags:</strong> Observation $t$ strictly predicts target $t+1$.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Regression Comparison
    st.markdown('<div class="section-title">1. Regression Models (Predicting PM2.5 in µg/m³)</div>', unsafe_allow_html=True)

    lr_test = reg["metrics"]["Linear Regression"]["test"]
    rf_test = reg["metrics"]["Random Forest Regressor"]["test"]

    col_eval_lr, col_eval_rf = st.columns(2)

    with col_eval_lr:
        st.markdown(f"""
        <div class="metric-grid-card" style="border-top: 4px solid #2563eb;">
            <div style="font-size:1.1rem; font-weight:800; color:#1e293b; margin-bottom:12px;">
                Linear Regression {'🏆 (Selected Best)' if reg['best_model_name'] == 'Linear Regression' else ''}
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#64748b; font-size:0.9rem;">Test R² Score:</span>
                <span style="font-weight:700; color:#0f172a;">{lr_test['R2']:.4f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#64748b; font-size:0.9rem;">Test RMSE:</span>
                <span style="font-weight:700; color:#0f172a;">{lr_test['RMSE']:.2f} µg/m³</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748b; font-size:0.9rem;">Test MAE:</span>
                <span style="font-weight:700; color:#0f172a;">{lr_test['MAE']:.2f} µg/m³</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_eval_rf:
        st.markdown(f"""
        <div class="metric-grid-card" style="border-top: 4px solid #0f766e;">
            <div style="font-size:1.1rem; font-weight:800; color:#1e293b; margin-bottom:12px;">
                Random Forest Regressor {'🏆 (Selected Best)' if reg['best_model_name'] == 'Random Forest Regressor' else ''}
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#64748b; font-size:0.9rem;">Test R² Score:</span>
                <span style="font-weight:700; color:#0f172a;">{rf_test['R2']:.4f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#64748b; font-size:0.9rem;">Test RMSE:</span>
                <span style="font-weight:700; color:#0f172a;">{rf_test['RMSE']:.2f} µg/m³</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748b; font-size:0.9rem;">Test MAE:</span>
                <span style="font-weight:700; color:#0f172a;">{rf_test['MAE']:.2f} µg/m³</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Actual vs Predicted Time-Series on Test Set
    st.markdown('<div class="section-title">Actual vs Predicted PM2.5 on Held-Out Test Set (400 Days)</div>', unsafe_allow_html=True)

    timestamps = [t[:10] for t in reg["test_timestamps"]]
    actual_test = reg["test_actuals"]
    pred_test = reg["best_predictions"]

    fig_eval_ts = go.Figure()
    fig_eval_ts.add_trace(go.Scatter(
        x=timestamps,
        y=actual_test,
        mode="lines",
        name="Actual Ground Truth",
        line=dict(color="#0f172a", width=2),
        hovertemplate="<b>Date:</b> %{x}<br><b>Actual:</b> %{y:.1f} µg/m³<extra></extra>"
    ))
    fig_eval_ts.add_trace(go.Scatter(
        x=timestamps,
        y=pred_test,
        mode="lines",
        name=f"Predicted ({reg['best_model_name']})",
        line=dict(color="#2563eb", width=2, dash="dot"),
        hovertemplate="<b>Date:</b> %{x}<br><b>Predicted:</b> %{y:.1f} µg/m³<extra></extra>"
    ))
    fig_eval_ts.update_layout(
        xaxis_title="Date (Chronological Test Partition: 2019–2020)",
        yaxis_title="PM2.5 (µg/m³)",
        template="plotly_white",
        height=390,
        margin=dict(l=40, r=40, t=20, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font={'family': 'Plus Jakarta Sans'}
    )
    st.plotly_chart(fig_eval_ts, use_container_width=True)

    # Scatter Plot & Residual Error Distribution
    col_scat1, col_scat2 = st.columns([1.2, 1])

    with col_scat1:
        fig_scat = go.Figure()
        fig_scat.add_trace(go.Scatter(
            x=actual_test,
            y=pred_test,
            mode="markers",
            marker=dict(color="#3b82f6", size=6, opacity=0.65, line=dict(color="#1d4ed8", width=0.5)),
            name="Daily Observations",
            hovertemplate="<b>Actual:</b> %{x:.1f} µg/m³<br><b>Predicted:</b> %{y:.1f} µg/m³<extra></extra>"
        ))
        max_val = max(max(actual_test), max(pred_test)) + 15
        fig_scat.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="#ef4444", dash="dash", width=2),
            name="Ideal Parity Fit (y = x)"
        ))
        fig_scat.update_layout(
            title="Goodness of Fit: Actual vs Predicted",
            xaxis_title="Actual PM2.5 (µg/m³)",
            yaxis_title="Predicted PM2.5 (µg/m³)",
            template="plotly_white",
            height=360,
            margin=dict(l=30, r=30, t=40, b=30),
            font={'family': 'Plus Jakarta Sans'}
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    with col_scat2:
        residuals = np.array(pred_test) - np.array(actual_test)
        fig_resid = px.histogram(
            residuals,
            nbins=35,
            title="Residual Error Distribution",
            labels={"value": "Residual Error (Predicted - Actual) in µg/m³"},
            color_discrete_sequence=["#0f766e"]
        )
        fig_resid.update_layout(
            height=360,
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            font={'family': 'Plus Jakarta Sans'}
        )
        st.plotly_chart(fig_resid, use_container_width=True)

    # 2. Classification Models Evaluation
    st.markdown('<div class="section-title">2. Classification Models (CPCB Category Bands)</div>', unsafe_allow_html=True)

    col_clf_dt, col_clf_rf = st.columns(2)

    with col_clf_dt:
        dt_m = clf["metrics"]["Decision Tree Classifier"]["test"]
        st.markdown(f"""
        <div class="metric-grid-card" style="border-top: 4px solid #8b5cf6;">
            <div style="font-size:1.1rem; font-weight:800; color:#1e293b; margin-bottom:12px;">
                Decision Tree Classifier {'🏆 (Selected Best)' if clf['best_model_name'] == 'Decision Tree Classifier' else ''}
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#64748b; font-size:0.9rem;">Test Accuracy:</span>
                <span style="font-weight:700; color:#0f172a;">{dt_m['Accuracy']*100:.1f}%</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#64748b; font-size:0.9rem;">Macro Precision:</span>
                <span style="font-weight:700; color:#0f172a;">{dt_m['Precision_Macro']:.3f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748b; font-size:0.9rem;">Macro F1-Score:</span>
                <span style="font-weight:700; color:#0f172a;">{dt_m['F1_Macro']:.3f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_clf_rf:
        rf_clf_m = clf["metrics"]["Random Forest Classifier"]["test"]
        st.markdown(f"""
        <div class="metric-grid-card" style="border-top: 4px solid #0284c7;">
            <div style="font-size:1.1rem; font-weight:800; color:#1e293b; margin-bottom:12px;">
                Random Forest Classifier {'🏆 (Selected Best)' if clf['best_model_name'] == 'Random Forest Classifier' else ''}
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#64748b; font-size:0.9rem;">Test Accuracy:</span>
                <span style="font-weight:700; color:#0f172a;">{rf_clf_m['Accuracy']*100:.1f}%</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#64748b; font-size:0.9rem;">Macro Precision:</span>
                <span style="font-weight:700; color:#0f172a;">{rf_clf_m['Precision_Macro']:.3f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748b; font-size:0.9rem;">Macro F1-Score:</span>
                <span style="font-weight:700; color:#0f172a;">{rf_clf_m['F1_Macro']:.3f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Confusion matrix
    best_clf_name = clf["best_model_name"]
    cm = clf["metrics"][best_clf_name]["test"]["confusion_matrix"]
    labels = clf["labels"]

    st.markdown(f'<div class="section-title">Confusion Matrix Heatmap ({best_clf_name})</div>', unsafe_allow_html=True)
    fig_cm = px.imshow(
        cm,
        x=labels,
        y=labels,
        labels=dict(x="Predicted Category", y="Actual Category", color="Count"),
        color_continuous_scale="Blues",
        text_auto=True
    )
    fig_cm.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=40, r=40, t=20, b=40),
        font={'family': 'Plus Jakarta Sans'}
    )
    st.plotly_chart(fig_cm, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 4: DATASET EXPLORER
# -----------------------------------------------------------------------------
elif nav_choice == "📁 Dataset Explorer":
    st.markdown("""
    <div class="hero-container" style="padding: 24px 30px; margin-bottom: 20px;">
        <div style="font-size: 1.85rem; font-weight: 800; color: #ffffff;">📁 Air Quality Dataset Explorer</div>
        <div style="font-size: 0.95rem; color: #cbd5e1; margin-top: 4px;">
            Inspect authentic historical ambient pollutant records from Delhi monitoring stations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df_clean is None:
        st.warning("Dataset not found at `data/air_quality_data.csv`.")
        st.stop()

    numeric_cols = [c for c in ["pm25", "pm10", "no2", "so2", "co", "o3"] if c in df_clean.columns]

    st.markdown('<div class="section-title">Statistical Summary Across Pollutants</div>', unsafe_allow_html=True)
    st.dataframe(df_clean[numeric_cols].describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown('<div class="section-title">Inter-Pollutant Correlation Heatmap</div>', unsafe_allow_html=True)
    corr_matrix = df_clean[numeric_cols].corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Pearson Correlation Matrix (Particulates vs Trace Gases)"
    )
    fig_corr.update_layout(height=420, template="plotly_white", margin=dict(l=30, r=30, t=40, b=30), font={'family': 'Plus Jakarta Sans'})
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown('<div class="section-title">Raw Observation Records Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_clean.head(50), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 5: UN SDG ALIGNMENT
# -----------------------------------------------------------------------------
elif nav_choice == "🎯 UN SDG Alignment":
    st.markdown("""
    <div class="hero-container" style="padding: 24px 30px; margin-bottom: 20px;">
        <div style="font-size: 1.85rem; font-weight: 800; color: #ffffff;">🎯 United Nations Sustainable Development Goals</div>
        <div style="font-size: 0.95rem; color: #cbd5e1; margin-top: 4px;">
            Demonstrating how Applied AI & Machine Learning support the UN 2030 Global Agenda.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_sdg1, col_sdg2 = st.columns(2)

    with col_sdg1:
        st.markdown("""
        <div class="sdg-box" style="background-color: #f0fdf4; border-color: #22c55e;">
            <div style="font-size: 1.45rem; font-weight: 800; color: #15803d; margin-bottom: 12px; display:flex; align-items:center; gap:8px;">
                🎯 SDG 3: Good Health and Well-Being
            </div>
            <div style="font-size: 0.92rem; color: #166534; line-height: 1.65;">
                <strong>Official Target 3.9:</strong> By 2030, substantially reduce the number of deaths and illnesses from hazardous chemicals and air, water and soil pollution.<br><br>
                <strong>AirAware AI Contributions:</strong>
                <ul>
                    <li><strong>Proactive Exposure Warnings:</strong> Forecasting PM2.5 24 hours in advance allows asthmatics and cardiac patients to plan medication and indoor stays.</li>
                    <li><strong>Health Impact Demystification:</strong> Converts complex numerical microgram measurements into straightforward health advisories.</li>
                    <li><strong>Vulnerable Group Shielding:</strong> Recommends specific protective actions for schools, elderly care, and outdoor workers.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_sdg2:
        st.markdown("""
        <div class="sdg-box" style="background-color: #eff6ff; border-color: #3b82f6;">
            <div style="font-size: 1.45rem; font-weight: 800; color: #1d4ed8; margin-bottom: 12px; display:flex; align-items:center; gap:8px;">
                🏙️ SDG 11: Sustainable Cities & Communities
            </div>
            <div style="font-size: 0.92rem; color: #1e40af; line-height: 1.65;">
                <strong>Official Target 11.6:</strong> By 2030, reduce the adverse per capita environmental impact of cities, including by paying special attention to air quality and municipal waste management.<br><br>
                <strong>AirAware AI Contributions:</strong>
                <ul>
                    <li><strong>Predictive Urban Management:</strong> Provides urban municipalities with empirical proof of temporal pollution persistence ($PM_{2.5}(t-1) \rightarrow PM_{2.5}(t)$).</li>
                    <li><strong>Seasonal Insight:</strong> Demonstrates severe winter thermal inversion spikes vs monsoon natural wet deposition washouts.</li>
                    <li><strong>Open-Source Scalability:</strong> Lightweight Python architecture deployable across municipal monitoring networks without proprietary software licenses.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>Academic & Ethical Disclaimer:</strong><br>
        AirAware AI is an educational machine learning demonstration prototype created for the AICTE–BharatCares–IBM SkillsBuild internship. 
        It does not diagnose, treat, or prevent any medical condition, nor does it issue statutory emergency declarations. Official air quality indexes 
        and health warnings must always be referenced from the respective statutory environmental authorities.
    </div>
    """, unsafe_allow_html=True)
