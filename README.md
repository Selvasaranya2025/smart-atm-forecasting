# smart-atm-forecasting
Smart ATM Forecasting System – ML + SHAP + Streamlit dashboard for ATM cash withdrawal demand prediction.
# Smart ATM Forecasting System (ML + SHAP + Streamlit Dashboard)

End-to-end **ATM cash demand forecasting** project built for a datathon challenge.

The system predicts:

- **Daily withdrawal amount (KWD)**
- **Daily withdrawal transaction count**

for every ATM over a **14-day horizon**, and provides **explainable ML** plus an **interactive Streamlit dashboard** for business users.

---

## 🔧 Tech Stack

- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `shap`, `matplotlib`, `plotly`  
- **Dashboard:** `streamlit`  
- **Type of Problem:** Time-series style regression (per ATM, per day)

---

## 📂 Data Sources

The project uses multiple datasets (ATM level + calendar + replenishment):

- `atm_transactions_train.csv` – historical daily ATM withdrawals (amount + count)  
- `atm_transactions_test.csv` – test period (dates & ATMs to forecast)  
- `atm_transactions_train_clean.csv` – cleaned version used for modeling  
- `atm_metadata.csv` – region, location type (branch/mall/off-site), install/decommission dates  
- `calendar.csv` – weekends, holidays, Ramadan, salary disbursement indicators  
- `cash_replenishment.csv` – starting cash, replenished amount, ending cash, cash-out flag  
- `atm_region_lookup.csv` – ATM → region mapping (backup)

> Note: Raw data is not included in the repo. Use your own or mock data with the same structure.

---

## 🧹 Task 1 – Data Cleaning & Baselines

### Cleaning steps

Implemented in `src/prepare_data_baseline.py`:

- Converted `dt` and `reported_dt` to proper datetime
- Removed:
  - exact duplicate rows
  - rows with `dup_flag = True`
- Aggregated multiple rows with same `(atm_id, dt)` by summing numeric fields
- Built **continuous daily series per ATM** (filled missing dates with 0 withdrawals)
- Clipped negative withdrawal amounts and counts to 0
- Merged region and metadata information

Output:  
`atm_transactions_train_clean.csv` + cleaned test panel

### Baseline models

Implemented in `src/train_task1_baseline.py`:

1. **Naive model** – last observed value per ATM  
2. **7-day Moving Average (MA7)** – mean of last 7 days  
3. **Simple Exponential Smoothing (SES)** – with α ≈ 0.3  

Validation:

- For each ATM, last 14 days used as validation
- Metric: **RMSE** per target (amount & count)

Result:

- **MA7** chosen as best-performing baseline.

---

## 🧠 Task 2 – Feature Engineering & ML Models

### Feature Engineering

Implemented in `src/train_task2_ml.py`:

**Calendar features**

- `dayofweek`, `month`, `quarter`, `year`
- `is_weekend`
- `is_public_holiday`
- `is_salary_disbursement`
- `is_ramadan`
- One-hot for `holiday_name` / special days

**ATM metadata features**

- `region`
- `location_type` (branch / mall / off-site)
- `atm_age_days` (days since installation)
- Optional flags for decommissioned ATMs

**Replenishment features**

- `starting_cash_kwd`
- `replenished_kwd`
- `ending_cash_kwd`
- `cashout_flag` (indicator if ATM ran out of cash)

**ATM historical statistics**

Computed per `atm_id`:

- `atm_mean_withdraw`
- `atm_std_withdraw`
- `atm_mean_count`
- `atm_std_count`

All categorical features are one-hot encoded using `pd.get_dummies`.

---

### ML Modeling

Two **RandomForestRegressor** models:

- Model A – predicts withdrawal amount (KWD)  
- Model B – predicts withdrawal transaction count  

Validation:

- Time-based split:
  - First 80% of days → training
  - Last 20% of days → validation
- Metric: **RMSE**

RandomForest chosen because:

- Handles non-linear relationships  
- Works well with mixed numeric + categorical features  
- Robust to noise and missing data  

Model artifacts saved as:  
`models/ml_models_task2.pkl`

This file contains:

- `model_amount`
- `model_count`
- `feature_columns` used during training

---

## 🔮 Prediction Script

`src/predict_task2.py`:

- Loads `ml_models_task2.pkl`
- Builds features for `atm_transactions_test.csv` using the **same logic** as training
- Aligns test features with `feature_columns`
- Outputs:

```text
predictions_task2.csv
  ├── dt
  ├── atm_id
  ├── predicted_withdrawn_kwd
  └── predicted_withdraw_count
🧾 Task 3 – Explainability & Insights

Implemented in src/analysis_task3.py:

RandomForest feature importance (global)

Permutation importance (on a sample)

SHAP TreeExplainer for both amount and count models

Outputs:

feature_importance_amount.csv / .png

feature_importance_count.csv / .png

shap_summary_amount.png / shap_summary_count.png

shap_bar_amount.png / shap_bar_count.png

region_level_summary.csv (aggregated metrics per region)

Key drivers (from explainability):

Salary cycle (salary disbursement days)

Weekends (Friday/Saturday)

Ramadan flag

Region & location type (mall / branch)

ATM age & historical demand profile

Replenishment behavior

🖥️ Streamlit Dashboard

Dashboard code: dashboard/vis.py

Features:

Historical ATM Activity

Total withdrawn amount, total transactions, average daily amount

Time-series plots by ATM / region / date range

Forecast Explorer (Baseline vs ML)

Load baseline (predictions.csv) or ML (predictions_task2.csv)

Filter by ATM, region, and date

Visualize:

predicted withdrawn amount (KWD)

predicted transaction count

Download filtered predictions as CSV

Scenario Simulation (What-If Analysis)

Salary-day multiplier (e.g., 1.3 → +30% demand)

Weekend multiplier (e.g., 1.1 → +10% demand)

Baseline vs scenario comparison

Total uplift and % change in demand
