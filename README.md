# AirAware AI — Air Quality Prediction System 🍃

**AirAware AI** is a machine learning web application that forecasts ambient **PM2.5 particulate concentration** using regression and categorizes air quality into health-relevant tiers using classification.

This project was built as an applied AI project for the **AICTE–BharatCares–IBM SkillsBuild Machine Learning and Applied AI Internship**, aligning with:
- **UN SDG 3:** Good Health and Well-Being (Target 3.9: Reduce illness and death from hazardous air pollution)
- **UN SDG 11:** Sustainable Cities and Communities (Target 11.6: Reduce per capita urban environmental impact)

---

## 📌 Project Overview & Objectives

Urban ambient air pollution, dominated by fine particulate matter ($PM_{2.5} \le 2.5\ \mu\text{m}$), poses severe respiratory and cardiovascular risks. Traditional monitoring stations report pollution retrospectively. **AirAware AI** introduces a predictive framework that:
1. **Predicts Next-Day ($t+1$) PM2.5 Numerical Concentration ($\mu\text{g/m}^3$):** Compares **Linear Regression** and **Random Forest Regressor** using time-lagged pollutant features and calendar seasonality.
2. **Classifies Air Quality into Standard Actionable Tiers:** Maps predictions into **Central Pollution Control Board (CPCB)** NAQI bands (*Good, Satisfactory, Moderate, Poor, Very Poor, Severe*) and evaluates direct tree-based classifiers (**Decision Tree** vs **Random Forest Classifier**).
3. **Guarantees Data Leakage Prevention:** Uses a strict chronological split (earlier 80% train, subsequent 20% test) with all lag and rolling features constructed backward in time, fitting imputers and scalers exclusively on the training split.
4. **Delivers an Interactive, Explainable Interface:** An intuitive Streamlit dashboard providing historical analytics, scenario presets, model evaluation diagnostics, and clear health advisories.

---

## 🛠️ Technology Stack

| Layer | Component | Version / Purpose |
| :--- | :--- | :--- |
| **Language** | Python | 3.11+ |
| **Web Interface** | Streamlit | Responsive, clean dashboard and prediction UI |
| **Data Processing** | Pandas, NumPy | Time-series manipulation, lag generation, range validation |
| **Machine Learning** | scikit-learn | Linear Regression, Random Forest Regressor & Classifier, Decision Tree |
| **Data Visualizations** | Plotly | Interactive time-series trends, scatter fits, and confusion matrices |
| **Model Persistence** | Joblib | Serialization of pipelines, scalers, and trained models |
| **Test Suite** | Pytest | Automated unit tests for preprocessing, non-leakage, and thresholds |

---

## 📊 Dataset Provenance & Attribution

- **Source:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change (MoEFCC), Govt. of India / OpenAQ.
- **Location:** Delhi Urban Ambient Monitoring Network.
- **Temporal Period:** January 1, 2015 to July 1, 2020 (2,007 verified daily observations).
- **Primary Predictors:**
  - PM2.5 lags ($t, t-1, t-2, t-6$) and 7-day rolling average.
  - Co-pollutants at time $t$: $PM_{10}$, $NO_2$, $SO_2$, $CO$, $O_3$.
  - Calendar attributes for horizon $t+1$: Month, Day of Week, Day of Year, Cyclical Month Encodings ($\sin$, $\cos$).
- **Target:** $PM_{2.5}$ concentration at time $t+1$ ($\mu\text{g/m}^3$).

---

## 📁 Project Structure

```
AirAwareAI/
├── app.py                      # Interactive Streamlit web application
├── train_models.py             # Standalone model training and artifact generator
├── evaluate_models.py          # Terminal model evaluation and performance reporter
├── src/
│   ├── __init__.py             # Package initializer
│   ├── air_quality.py          # CPCB/EPA category thresholds and health advisories
│   ├── data_loader.py          # Data ingestion, schema standardization, and physical validation
│   ├── preprocessing.py        # Leakage-free lag creation, chronological split, pipeline
│   ├── regression.py           # Linear Regression & Random Forest Regressor
│   └── classification.py       # Decision Tree & Random Forest Classifier
├── data/
│   ├── air_quality_data.csv    # Historical Delhi air quality observations
│   └── README.md               # Dataset documentation and custom setup guide
├── models/
│   ├── preprocessor.joblib     # Fitted imputer and scaler pipeline
│   ├── best_regressor.joblib   # Selected best regression model
│   ├── all_regressors.joblib   # All trained regression models
│   ├── best_classifier.joblib  # Selected best classification model
│   ├── all_classifiers.joblib # All trained classification models
│   └── training_summary.joblib # Test metrics, actuals, and confusion matrices
├── tests/
│   ├── __init__.py             # Test suite initializer
│   ├── test_preprocessing.py   # Leakage prevention & feature pipeline tests
│   └── test_thresholds.py      # Threshold boundary & edge case tests
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore rules
```

---

## 🚀 Quickstart & Setup (Windows / PowerShell)

### 1. Create and Activate Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the Automated Unit Tests
```powershell
pytest tests -v
```

### 4. Train the Machine Learning Models
```powershell
python train_models.py
```

### 5. Evaluate Model Performance
```powershell
python evaluate_models.py
```

### 6. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📈 Model Evaluation & Test Results

Evaluated on **400 held-out chronological test days** (May 28, 2019 to June 30, 2020) without lookahead bias:

### 1. Regression Comparison ($PM_{2.5}$ in $\mu\text{g/m}^3$)
| Model | Split | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression** | Train / **Test** | 24.80 / **23.25** | 37.93 / **38.65** | 0.7879 / **0.7678** |
| **Random Forest Regressor** | Train / **Test** | 12.15 / **23.54** | 19.85 / **40.14** | 0.9419 / **0.7496** |

- **Outcome:** Linear Regression demonstrates superior out-of-sample generalization with lower Test RMSE ($38.65\ \mu\text{g/m}^3$) and higher Test $R^2$ ($0.768$), avoiding overfitting observed in deep tree ensembles on cyclical timeseries.

### 2. Classification Comparison (CPCB Air Quality Tiers)
| Classifier | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree Classifier** | **56.25%** | **0.4955** | **0.4838** | **0.4832** |
| **Random Forest Classifier** | 56.00% | 0.5596 | 0.4442 | 0.4467 |

---

## 🛡️ Data Leakage Prevention Principles

This project adheres strictly to time-series machine learning standards:
1. **Chronological Splitting:** Observations are never shuffled. The earliest 80% (2015–2019) is used for training; the subsequent 20% (2019–2020) is held out for testing.
2. **Backward-Looking Lag Construction:** Features at step $t$ ($PM_{2.5}(t), PM_{2.5}(t-1), \dots$) use past values exclusively to predict the target at step $t+1$.
3. **Fitted Preprocessing Isolation:** Median imputation and standard scaling are fitted **strictly on the training split** and applied downstream to the test set and single-sample web inputs.

---

## 🎯 United Nations SDG Alignment

- **SDG 3: Good Health and Well-Being:** Provides accessible, actionable health advisories categorized by PM2.5 exposure levels to protect vulnerable populations (asthmatics, elderly, children).
- **SDG 11: Sustainable Cities and Communities:** Equips citizens and municipal stakeholders with predictive insights into urban air quality dynamics and seasonal smog episodes.

---

## ⚠️ Academic & Ethical Disclaimer

AirAware AI is an educational demonstration prototype developed for academic evaluation. Predictions are statistical machine learning estimates based on historical measurements and do not constitute statutory air quality declarations, medical advice, or regulatory warnings. Official AQI bulletins must always be obtained from statutory environmental authorities (e.g. CPCB / US EPA).

