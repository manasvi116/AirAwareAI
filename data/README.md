# AirAware AI — Dataset Documentation

## 1. Overview and Provenance
- **Dataset Title:** Delhi Ambient Air Quality Historical Time-Series
- **Original Source:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change (MoEFCC), Government of India / OpenAQ Repository.
- **Monitoring Location:** Delhi Urban Ambient Monitoring Network, India.
- **Temporal Coverage:** January 1, 2015 to July 1, 2020 (Daily aggregates, 2,007 verified observations).
- **Primary Target:** `PM2.5` (Fine Particulate Matter $\le 2.5\ \mu\text{m}$) in $\mu\text{g/m}^3$.

## 2. Variables & Measurement Units
| Variable | Description | Units | Physical Bounds |
| :--- | :--- | :--- | :--- |
| `Date` / `timestamp` | Daily observation timestamp | YYYY-MM-DD | 2015-01-01 to 2020-07-01 |
| `PM2.5` | Particulate matter $\le 2.5\ \mu\text{m}$ | $\mu\text{g/m}^3$ | 0.0 – 1500.0 |
| `PM10` | Particulate matter $\le 10\ \mu\text{m}$ | $\mu\text{g/m}^3$ | 0.0 – 2000.0 |
| `NO2` | Nitrogen Dioxide | $\mu\text{g/m}^3$ | 0.0 – 600.0 |
| `SO2` | Sulfur Dioxide | $\mu\text{g/m}^3$ | 0.0 – 600.0 |
| `CO` | Carbon Monoxide | $\text{mg/m}^3$ | 0.0 – 100.0 |
| `O3` | Ground-level Ozone | $\mu\text{g/m}^3$ | 0.0 – 600.0 |

## 3. Data Cleaning & Validation
- **Missing Value Handling:** Missing records in pollutant time-series are imputed using training-split median values.
- **Range Enforcement:** Negative readings and extreme non-physical spikes are filtered.
- **Deduplication:** Chronological ordering is enforced with timestamp deduplication.

## 4. Custom Dataset Setup
To use a custom dataset:
1. Place your CSV file in `data/air_quality_data.csv`.
2. Ensure columns include a timestamp (`Date` or `timestamp`) and a target (`PM2.5` or `pm25`).
3. Retrain the models using `python train_models.py`.

