# 📦 Retail Demand Forecasting System

> End-to-end machine learning system that predicts daily item-level sales across multiple store locations - built with LightGBM, FastAPI, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Best%20Model-brightgreen)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)

---

##  Live Links

| What | Link |
|---|---|
|  Streamlit Dashboard | `https://nabinkatwal-retail-forecast.streamlit.app` |
|  FastAPI Docs (Swagger) | `https://retail-demand-api.onrender.com/docs` |
|  GitHub Repository | `https://github.com/cyrolox123/retail-demand-forecasting` |

---

##  What This Project Does

This system forecasts **how many units of each product will be sold** at each store location on any given day. Accurate demand forecasting helps businesses:

- **Cut food/product waste** by not over-preparing
- **Prevent stockouts** by not under-preparing
- **Plan staff and logistics** around expected demand

The project replicates a real-world production forecasting pipeline - from raw data to a live web API that any system or dashboard can query.

---

##  Project Structure

```
retail-demand-forecasting/
│
├──  Notebooks
│   ├── machine_learning.ipynb          ← Data loading, EDA, feature engineering, ML models
│   ├── time_series.ipynb          ← Statistical models (ARIMA, SARIMA, SARIMAX, Prophet)
│   └── deep_learning.ipynb          ← Deep learning models (LSTM, GRU, Bidirectional LSTM)
│
├── main.py          ← FastAPI backend (REST API server)
│                     
├── app.py           ← Streamlit frontend (interactive dashboard)
│
├── Data
│   ├── train.csv                ← 913,000 rows of historical daily sales (2013–2017)
│   └── test.csv                 ← 45,000 rows to forecast (2018)
│
├── saved_models/
│   └── 
│       ├── lightgbm.pkl         ← Best model (used by the API)
│       ├── feature_cols.json    ← Exact feature list the model was trained on
│       ├── xgboost_model.json
│       ├── catboost_model.cbm
│       ├── random_forest.pkl
│       ├── adaboost.pkl
│       ├── gradient_boosting.pkl
│       ├── linear_regression.pkl
│       ├── arima_store1_item1.pkl
│       ├── sarima_store1_item1.pkl
│       ├── sarimax_store1_item1.pkl
│       ├── prophet_store1_item1.pkl
│       ├── lstm_model.keras
│       ├── gru_model.keras
│       ├── bilstm_model.keras
│       ├── mv_lstm_model.keras
│       ├── lstm_scaler.pkl      ← MinMaxScaler for LSTM input
│       ├── mv_scaler_X.pkl      ← Feature scaler for multivariate LSTM
│       ├── mv_scaler_y.pkl      ← Target scaler for multivariate LSTM
│       └── mv_lstm_meta.json    ← Lookback window and feature list for multivariate LSTM
│
├── Outputs
│   └── submission_lightgbm.csv  ← Kaggle-format predictions on test set
│
├── requirements.txt             ← All Python dependencies
└── README.md                    ← This file
```

---

## 📊 Dataset

**Source:** [Kaggle - Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only)

| Column | Description |
|---|---|
| `date` | The date of the sale (`YYYY-MM-DD`) |
| `store` | Store ID - 10 unique stores |
| `item` | Item ID - 50 unique items per store |
| `sales` | Number of units sold on that date at that store |

- **Training data:** 1 Jan 2013 → 31 Dec 2017 (5 years, 913,000 rows)
- **Test data:** 1 Jan 2018 → 31 Mar 2018 (3 months to forecast)
- **Total series:** 10 stores × 50 items = **500 individual time series**
- **No missing values**, no store closures, no holiday effects in raw data

---

##  Models Trained

### Notebook 1 - Machine Learning Models

These models treat the problem as a **tabular regression task**. Every day becomes one row with engineered features. All models are trained on the full dataset across all stores and items simultaneously (called a **global forecasting model**).

| Model | What it is | MAE | RMSE | R² |
|---|---|---|---|---|
| **LightGBM** | Fast gradient boosting with leaf-wise tree growth | **0.47** | **0.64** | **0.9995** |
| CatBoost | Gradient boosting with built-in categorical handling | 0.56 | 0.72 | 0.9994 |
| XGBoost | Classic gradient boosting (column-wise trees) | 0.59 | 0.77 | 0.9993 |
| Gradient Boosting | Scikit-learn's gradient boosting | - | - | - |
| Random Forest | Ensemble of independent decision trees | 0.77 | 1.16 | 0.9983 |
| AdaBoost | Boosting that focuses on hard examples | 6.26 | 8.09 | 0.9192 |
| Linear Regression | Simple baseline - straight-line fit | baseline | baseline | - |

**Why LightGBM wins:** It handles large datasets efficiently, supports early stopping, and naturally captures the non-linear seasonal patterns in demand data.

### Notebook 2 - Statistical / Classical Models

These models are **trained per store-item series** (one model per series). They work directly on the raw time series without feature engineering.

| Model | What it is |
|---|---|
| **ARIMA** | AutoRegressive Integrated Moving Average - captures trend and autocorrelation |
| **SARIMA** | ARIMA + Seasonal component - adds weekly/yearly patterns |
| **SARIMAX** | SARIMA + eXogenous variables - adds external signals (day of week, month) |
| **Prophet** | Facebook's model - handles holidays, trend changes, multiple seasonalities automatically |

> These models are illustrated on Store 1, Item 1 but include a scaling template to train all 500 series.

### Notebook 3 - Deep Learning Models

These models learn **temporal patterns from raw sequences** of past sales.

| Model | What it is |
|---|---|
| **LSTM** | Long Short-Term Memory - remembers long-range patterns through gating |
| **GRU** | Gated Recurrent Unit - lighter and faster than LSTM, similar accuracy |
| **Bidirectional LSTM** | Reads the sequence forward and backward simultaneously |
| **Multivariate LSTM** | LSTM with 15 engineered features as input (lags, rolling stats, calendar) |

All deep learning models use:
- `MinMaxScaler` to scale inputs to [0, 1]
- `EarlyStopping` to prevent overfitting
- `ReduceLROnPlateau` to lower the learning rate when progress stalls
- `ModelCheckpoint` to save the best weights

---

## Feature Engineering

The most important part of the pipeline. Raw date + store + item → 50+ predictive features.

### Calendar Features
| Feature | What it captures |
|---|---|
| `year`, `month`, `day` | Basic date components |
| `dayofweek` | 0=Monday … 6=Sunday |
| `dayofyear` | 1–365, captures annual position |
| `weekofyear` | Week number 1–52 |
| `quarter` | Q1–Q4 |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_month_start/end` | Payday effects at start/end of month |

### Cyclical Encoding
Converts periodic features into sine/cosine pairs so the model understands that December (month 12) is close to January (month 1) - not far apart.

| Feature | Why |
|---|---|
| `month_sin`, `month_cos` | Circular encoding of month |
| `dayofweek_sin`, `dayofweek_cos` | Circular encoding of weekday |
| `dayofyear_sin`, `dayofyear_cos` | Circular encoding of annual position |

### Lag Features
Past sales values used as predictors. The model learns "what was sold 7 days ago" to predict today.

| Feature | Meaning |
|---|---|
| `lag_7` | Sales exactly 7 days ago (same weekday last week) |
| `lag_14` | Sales 2 weeks ago |
| `lag_21` | Sales 3 weeks ago |
| `lag_28` | Sales 4 weeks ago |
| `lag_90` | Sales ~3 months ago (quarterly pattern) |
| `lag_180` | Sales ~6 months ago |
| `lag_365` | Sales exactly 1 year ago (strongest signal) |

### Rolling Statistics
Computed over a sliding window of past sales - captures recent trends and volatility.

| Feature | Meaning |
|---|---|
| `rolling_mean_7` | Average sales over past 7 days |
| `rolling_mean_14` | Average sales over past 14 days |
| `rolling_mean_30` | Average sales over past 30 days |
| `rolling_mean_90` | Average sales over past 90 days |
| `rolling_std_7/14/30/90` | Sales volatility over each window |
| `rolling_max_7/14/30/90` | Peak sales in each window |
| `rolling_min_7/14/30/90` | Lowest sales in each window |

### Trend Features
| Feature | Meaning |
|---|---|
| `expanding_mean` | Average of all past sales (growing window) |
| `sales_diff_1` | Change from yesterday (day-over-day delta) |
| `sales_diff_7` | Change from same day last week |

> **Important:** All lag and rolling features use `.shift(1)` before computing to prevent data leakage - the model never sees future sales during training.

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/cyrolux123/Retail-Demand-Forecasting.git
cd retail-demand-forecasting
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebooks (train models)

Open Jupyter and run in order:

```bash
jupyter notebook
```

- Run `machine_learning.ipynb` first - this trains all ML models and saves them to `saved_models/`
- Run `time_series.ipynb` - trains ARIMA, SARIMA, SARIMAX, Prophet
- Run `deep_learning.ipynb` - trains LSTM, GRU, Bidirectional LSTM, Multivariate LSTM

> After running machine_learning, the `saved_models/` folder will contain all the files needed for the API.

### 5. Start the FastAPI backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` to see the interactive API documentation.

### 6. Start the Streamlit dashboard

Open a second terminal (keep FastAPI running in the first):

```bash
streamlit run app.py
```

Visit `http://localhost:8501` to see the dashboard.

---

## API Endpoints

The FastAPI backend exposes these endpoints:

### `GET /health`
Check if the API is running and the model is loaded.

```json
{
  "status": "ok",
  "model_loaded": true,
  "n_features": 52,
  "timestamp": "2026-01-15T10:30:00"
}
```

### `GET /model-info`
Returns the model type, number of features, and the full feature list.

### `POST /predict`
Predict sales for a single store, item, and date.

**Request:**
```json
{
  "store": 1,
  "item": 5,
  "date": "2018-03-15"
}
```

**Response:**
```json
{
  "store": 1,
  "item": 5,
  "date": "2018-03-15",
  "predicted_sales": 47,
  "model_used": "LightGBM"
}
```

### `POST /predict-range`
Forecast every day in a date range for one store-item pair.

**Request:**
```json
{
  "store": 2,
  "item": 10,
  "start_date": "2018-01-01",
  "end_date": "2018-03-31"
}
```

**Response:** Returns `total_days`, `total_predicted_sales`, and a list of `{date, predicted_sales}` for every day.

### `POST /batch-predict`
Send multiple store-item-date rows as JSON, get predictions for all.

**Request:**
```json
{
  "rows": [
    {"store": 1, "item": 1, "date": "2018-01-01"},
    {"store": 2, "item": 5, "date": "2018-01-01"}
  ]
}
```

### `POST /upload-predict`
Upload a CSV file with columns `store, item, date` - returns a CSV with predictions added.

### `GET /store-summary/{store_id}?days=30`
Forecast all 50 items for a store over the next N days. Returns item-level totals and averages.

---

##  Streamlit Dashboard Pages

| Page | What you can do |
|---|---|
| Overview | Model performance comparison, R² chart, project summary |
| Single Prediction | Pick a store, item, date → see predicted sales with a gauge chart |
| Date Range Forecast | Forecast a full date range, view line chart + weekly aggregation, download CSV |
| Store Dashboard | All-item forecast for a store - bar charts, top 10 / bottom 10 items |
| CSV Upload | Upload your own CSV, get predictions back, download results |
| Model Info | Full feature list, feature category pie chart, model metadata |

---

## MLflow Experiment Tracking

All ML model runs are logged with MLflow. To view the experiment dashboard:

```bash
mlflow ui
```

Visit `http://localhost:5000` to compare all model runs side by side - metrics, parameters, and saved artifacts.

The experiment is named `Retail_Demand_Forecasting`. Each run logs:
- MAE, RMSE, SMAPE, R²
- The trained model artifact
- Run name matching the model (e.g. `LightGBM`, `XGBoost`)

---

## Validation Strategy

**Time-based split** - never shuffle time series data randomly. Shuffling would cause data leakage (future data leaking into training).

```
Training:   2013-01-01  →  2017-09-30   (all but last 3 months)
Validation: 2017-10-01  →  2017-12-31   (last 3 months of training data)
Test:       2018-01-01  →  2018-03-31   (Kaggle test set, no labels)
```

Rows where `lag_365` is NaN (the first ~365 days per series) are dropped before training - the model needs at least one full year of history to compute all lag features.

---

## Metrics Explained

| Metric | Formula | What it means |
|---|---|---|
| **MAE** | mean(|actual − predicted|) | Average error in sales units - easy to interpret |
| **RMSE** | √mean((actual − predicted)²) | Penalises large errors more heavily than MAE |
| **R²** | 1 − (SS_res / SS_tot) | How much variance the model explains. 1.0 = perfect, 0 = no better than mean |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data processing | `pandas`, `numpy` | Data loading, cleaning, feature engineering |
| Visualisation | `matplotlib`, `seaborn`, `plotly` | EDA charts and dashboard plots |
| ML models | `scikit-learn` | Linear Regression, Random Forest, AdaBoost, Gradient Boosting |
| Boosting | `lightgbm`, `xgboost`, `catboost` | State-of-the-art gradient boosting models |
| Statistical | `statsmodels` | ARIMA, SARIMA, SARIMAX |
| Forecasting | `prophet` | Facebook Prophet |
| Deep learning | `tensorflow` / `keras` | LSTM, GRU, Bidirectional LSTM |
| Model saving | `joblib`, `pickle` | Persist trained models to disk |
| Experiment tracking | `mlflow` | Log metrics, compare runs, version models |
| API | `fastapi`, `uvicorn` | Production REST API with auto-generated docs |
| Dashboard | `streamlit` | Interactive web frontend |
| Deployment - API | Render.com | Free cloud hosting for FastAPI |
| Deployment - UI | Streamlit Cloud | Free cloud hosting for Streamlit |

---

## Author

**Nabin Katwal**
Data Scientist · Kathmandu, Nepal

[![LinkedIn](https://img.shields.io/badge/LinkedIn-nabinkatwal-blue?logo=linkedin)](https://linkedin.com/in/nabinkatwal)
[![GitHub](https://img.shields.io/badge/GitHub-cyrolox123-black?logo=github)](https://github.com/cyrolox123)
[![Portfolio](https://img.shields.io/badge/Portfolio-nabinkatwal.com.np-orange)](https://nabinkatwal.com.np)
[![Email](https://img.shields.io/badge/Email-katwalnk369@gmail.com-red?logo=gmail)](mailto:katwalnk369@gmail.com)

---

---

*Built as a portfolio project demonstrating end-to-end ML engineering - from raw time series data to a live production forecasting API.*
