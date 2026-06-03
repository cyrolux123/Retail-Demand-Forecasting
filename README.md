# 📦 Retail Demand Forecasting System

An end-to-end Machine Learning and Deep Learning forecasting platform for predicting daily retail sales across multiple stores and products.

This project uses the Kaggle **Store Item Demand Forecasting Challenge** dataset and demonstrates the complete MLOps workflow:

* Data Analysis & Feature Engineering
* Machine Learning Forecasting Models
* Deep Learning Time-Series Models
* Model Comparison & Evaluation
* FastAPI Prediction Service
* Streamlit Interactive Dashboard
* Batch Forecasting via CSV Upload

---

# 🚀 Project Overview

Retailers need accurate demand forecasts to optimize:

* Inventory Management
* Supply Chain Planning
* Warehouse Operations
* Procurement Decisions
* Revenue Forecasting

This project predicts daily sales for:

* 10 Stores
* 50 Items
* 5 Years of Daily Sales Data
* 500 Individual Time Series

The final production model is a **LightGBM Regressor**, deployed through a FastAPI backend and visualized with a Streamlit dashboard.

---

# 📊 Dataset

### Store Item Demand Forecasting Challenge

Dataset contains:

| Column | Description          |
| ------ | -------------------- |
| date   | Sales date           |
| store  | Store ID (1–10)      |
| item   | Item ID (1–50)       |
| sales  | Daily sales quantity |

### Dataset Statistics

| Metric     | Value                   |
| ---------- | ----------------------- |
| Stores     | 10                      |
| Items      | 50                      |
| Series     | 500                     |
| Records    | 913,000+                |
| Date Range | 2013-01-01 → 2017-12-31 |

---

# 🏗 Project Architecture

```text
                    ┌─────────────────┐
                    │   Streamlit UI  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   FastAPI API   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ LightGBM Model  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Sales Forecasts │
                    └─────────────────┘
```

---

# 📂 Project Structure

```text
Retail-Demand-Forecasting/
│
├── notebooks/
│   ├── notebook_1_machine_learning.ipynb
│   ├── notebook_2_eda_feature_engineering.ipynb
│   ├── notebook_3_deep_learning.ipynb
│
├── saved_models/
│   ├── lightgbm.pkl
│   ├── feature_cols.json
│   ├── lstm_model.keras
│   ├── gru_model.keras
│   ├── bilstm_model.keras
│   ├── mv_lstm_model.keras
│
├── main.py
├── app.py
│
├── train.csv
├── test.csv
│
├── requirements.txt
├── README.md
│
└── screenshots/
```

---

# 🔬 Feature Engineering

The forecasting model uses advanced time-series features.

## Calendar Features

```python
year
month
day
dayofweek
dayofyear
weekofyear
quarter
is_weekend
is_month_start
is_month_end
```

---

## Cyclical Features

```python
month_sin
month_cos
dayofweek_sin
dayofweek_cos
dayofyear_sin
dayofyear_cos
```

These preserve seasonality information.

---

## Lag Features

```python
lag_7
lag_14
lag_21
lag_28
lag_90
lag_180
lag_365
```

These allow the model to learn historical demand patterns.

---

## Rolling Statistics

```python
rolling_mean_7
rolling_std_7
rolling_max_7
rolling_min_7

rolling_mean_14
rolling_std_14

rolling_mean_30
rolling_std_30

rolling_mean_90
rolling_std_90
```

---

## Trend Features

```python
expanding_mean
sales_diff_1
sales_diff_7
```

---

# 🤖 Machine Learning Models

Several models were trained and compared.

| Model             | MAE      | RMSE     | SMAPE    |
| ----------------- | -------- | -------- | -------- |
| LightGBM          | 0.4665   | 0.6368   | 1.10%    |
| CatBoost          | 0.5610   | 0.7240   | 1.40%    |
| XGBoost           | 0.5879   | 0.7745   | 1.39%    |
| Random Forest     | 0.7668   | 1.1638   | 2.28%    |
| Linear Regression | Baseline | Baseline | Baseline |

🏆 **Best Model: LightGBM**

---

# 🧠 Deep Learning Models

The project also includes neural-network forecasting models.

## Models Implemented

### LSTM

```python
LSTM → Dropout → BatchNorm
LSTM → Dense → Output
```

### GRU

```python
GRU → Dropout → BatchNorm
GRU → Dense → Output
```

### Bidirectional LSTM

```python
BiLSTM → Dropout
BiLSTM → Dense
```

### Multivariate LSTM

Features used:

* Calendar Features
* Lag Features
* Rolling Features
* Cyclical Features

---

# 📈 Evaluation Metrics

Models are evaluated using:

### MAE

Mean Absolute Error

```text
Average absolute prediction error
```

---

### RMSE

Root Mean Squared Error

```text
Penalizes larger forecasting mistakes
```

---

### SMAPE

Symmetric Mean Absolute Percentage Error

```text
Percentage-based forecasting accuracy metric
```

---

### R² Score

```text
Explained variance of the model
```

---

# ⚡ FastAPI Backend

The API serves predictions using the trained LightGBM model.

---

## Start API

```bash
pip install fastapi uvicorn lightgbm pandas numpy joblib

uvicorn main:app --reload
```

API runs at:

```text
http://localhost:8000
```

---

## Interactive Documentation

Swagger UI:

```text
http://localhost:8000/docs
```

ReDoc:

```text
http://localhost:8000/redoc
```

---

# 🔗 API Endpoints

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

## Model Information

```http
GET /model-info
```

Returns:

* Model metadata
* Feature list
* Supported stores/items

---

## Single Prediction

```http
POST /predict
```

Request:

```json
{
  "store": 1,
  "item": 5,
  "date": "2018-01-01"
}
```

Response:

```json
{
  "predicted_sales": 38
}
```

---

## Date Range Forecast

```http
POST /predict-range
```

Returns forecasts for multiple days.

---

## Batch Prediction

```http
POST /batch-predict
```

Predict thousands of rows in a single request.

---

## CSV Upload

```http
POST /upload-predict
```

Upload:

```csv
store,item,date
1,1,2018-01-01
1,2,2018-01-01
```

Returns:

```csv
store,item,date,predicted_sales
1,1,2018-01-01,52
```

---

## Store Dashboard Summary

```http
GET /store-summary/{store_id}
```

Returns:

* Total store forecast
* Item-level forecasts
* Top selling products

---

# 🎨 Streamlit Dashboard

Interactive frontend built using Streamlit.

---

## Run Dashboard

```bash
pip install streamlit requests plotly

streamlit run app.py
```

---

## Dashboard Pages

### 🏠 Overview

Displays:

* Project summary
* Model performance
* Metrics comparison

---

### 🔮 Single Prediction

Predict sales for:

* Store
* Item
* Date

---

### 📅 Date Range Forecast

Forecast sales over a custom date range.

Features:

* Interactive charts
* Weekly aggregation
* CSV export

---

### 📊 Store Dashboard

Store-wide analytics:

* Forecast all 50 items
* Top 10 products
* Bottom 10 products
* Download reports

---

### 📤 CSV Upload

Batch forecasting:

* Upload CSV
* Predict thousands of rows
* Download results

---

### ℹ️ Model Info

Displays:

* Feature list
* Feature categories
* Model metadata

---

# 📷 Dashboard Screenshots

Add screenshots here:

```text
screenshots/
├── overview.png
├── single_prediction.png
├── range_forecast.png
├── store_dashboard.png
├── csv_upload.png
```

---

# 🛠 Installation

Clone repository:

```bash
git clone https://github.com/yourusername/retail-demand-forecasting.git

cd retail-demand-forecasting
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# requirements.txt

```text
pandas
numpy
scikit-learn
lightgbm
xgboost
catboost
tensorflow
keras
matplotlib
seaborn
plotly
streamlit
fastapi
uvicorn
joblib
python-multipart
requests
```

---

# ▶ Running the Full Application

### Step 1

Train models

```bash
jupyter notebook
```

Run training notebooks.

---

### Step 2

Start FastAPI

```bash
uvicorn main:app --reload
```

---

### Step 3

Start Streamlit

```bash
streamlit run app.py
```

---

### Step 4

Open Dashboard

```text
http://localhost:8501
```

---

# Future Improvements

* Docker Deployment
* CI/CD Pipeline
* AWS Deployment
* Real-time Forecasting
* Model Monitoring
* Historical Data Upload API
* Forecast Explainability (SHAP)
* Prophet Comparison
* Transformer-Based Forecasting

---

# Key Skills Demonstrated

### Machine Learning

* LightGBM
* XGBoost
* CatBoost
* Random Forest

### Deep Learning

* LSTM
* GRU
* Bidirectional LSTM
* Multivariate LSTM

### Data Science

* Time Series Forecasting
* Feature Engineering
* Model Evaluation
* Data Visualization

### Backend Development

* FastAPI
* REST APIs
* Model Serving

### Frontend Development

* Streamlit
* Plotly Dashboards

### Software Engineering

* Modular Design
* API Development
* Model Deployment
* Production-Ready Architecture

---

# Author

**Nabin Katwal**

Data Science | Machine Learning | Time Series Forecasting

Portfolio Project: Retail Demand Forecasting System

---


#   R e t a i l - D e m a n d - F o r e c a s t i n g  
 