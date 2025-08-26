# time-series-weekly-sales-forecast-fmcg
A machine learning pipeline for forecasting weekly product sales using advanced time series modeling.
# 📈 Weekly Sales Forecasting with CatBoost

This project demonstrates how to forecast **weekly product sales (`units_sold`)** using advanced time series modeling techniques.  
The model leverages **feature engineering** and **CatBoost regression** to capture short- and long-term seasonal trends in sales data.

---

## 🚀 Project Workflow

### 1. Data Preparation
- Parsed `date` column into datetime format (`YYYY-MM-DD`).
- Aggregated daily sales into **weekly sales**.
- Created a `week` column representing the start of each week.

### 2. Feature Engineering
- **Lag Features**: `lag_1, lag_2, lag_3, lag_4, lag_12, lag_26, lag_52, lag_104`
- **Rolling Statistics**: 4-week and 12-week rolling mean & standard deviation
- **Calendar Features**: Month, Quarter, Week-of-Year, Year

### 3. Modeling
- Implemented **CatBoost Regressor** with:
  - `iterations=1000`
  - `depth=8`
  - `learning_rate=0.05`
  - `loss_function='RMSE'`
- Trained using **TimeSeriesSplit cross-validation** to prevent data leakage.

### 4. Evaluation & Visualization
- Compared **actual vs. predicted** weekly sales.
- Decomposed time series into **trend, seasonality, and residuals**.
- Visualized results using Matplotlib.

---


## 🔧 Requirements

Install dependencies with:

```bash
pip install pandas numpy matplotlib scikit-learn catboost statsmodels
