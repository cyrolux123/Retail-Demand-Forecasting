# =============================================================================
# Streamlit Frontend — Retail Demand Forecasting Dashboard
# File: app.py
# Author: Nabin Katwal | Retail Demand Forecasting Portfolio Project
#
# HOW TO RUN (FastAPI must be running first):
#   pip install streamlit requests pandas plotly
#   streamlit run app.py
#
# Make sure main.py (FastAPI) is running at http://localhost:8000
# =============================================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import json

# 
# CONFIG

USE_RENDER = True

if USE_RENDER:
    API_URL = "https://retail-demand-api.onrender.com"
else:
    API_URL = "http://localhost:8000" 

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 
# HELPER: API CALLS
# 

def check_api_health():
    """Returns True if FastAPI is reachable."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200, response.json()
    except Exception:
        return False, {}


def get_model_info():
    """Fetch model metadata from the API."""
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        return response.json()
    except Exception:
        return {}


def predict_single(store, item, date):
    """Call /predict for one row."""
    payload = {"store": store, "item": item, "date": str(date)}
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def predict_range(store, item, start_date, end_date):
    """Call /predict-range for a date range."""
    payload = {
        "store":      store,
        "item":       item,
        "start_date": str(start_date),
        "end_date":   str(end_date)
    }
    response = requests.post(f"{API_URL}/predict-range", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def batch_predict_json(rows: list):
    """Call /batch-predict with a list of {store, item, date} dicts."""
    payload = {"rows": rows}
    response = requests.post(f"{API_URL}/batch-predict", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def upload_csv_predict(file_bytes, filename):
    """Call /upload-predict with a CSV file."""
    files    = {"file": (filename, file_bytes, "text/csv")}
    response = requests.post(f"{API_URL}/upload-predict", files=files, timeout=120)
    response.raise_for_status()
    return response.content   # returns CSV bytes


def get_store_summary(store_id, days):
    """Call /store-summary/{store_id}."""
    response = requests.get(
        f"{API_URL}/store-summary/{store_id}",
        params={"days": days},
        timeout=60
    )
    response.raise_for_status()
    return response.json()

# 
# SIDEBAR
# 
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("Demand Forecasting")
    st.caption("LightGBM · Store Item Demand · Nabin Katwal")

    st.divider()

    # API health badge
    is_healthy, health_data = check_api_health()
    if is_healthy:
        st.success(" API Online")
    else:
        st.error(" API Offline — start FastAPI first")
        st.code("uvicorn main:app --reload", language="bash")

    st.divider()

    # Navigation
    page = st.radio(
        "Navigate",
        options=[
            " Overview",
            " Single Prediction",
            " Date Range Forecast",
            " Store Dashboard",
            " CSV Upload"
        ]
    )

# 
# HELPER: Plotly chart for forecasts
# 
def plot_forecast(df: pd.DataFrame, title: str):
    """Line chart of predicted sales over time."""
    fig = px.line(
        df,
        x="date",
        y="predicted_sales",
        title=title,
        labels={"predicted_sales": "Predicted Sales", "date": "Date"},
        markers=True,
        template="plotly_dark"
    )
    fig.update_traces(line_color="#00d4ff", marker_color="#ff6b6b")
    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        title_font_size=16,
        hovermode="x unified"
    )
    return fig


def plot_bar_items(df: pd.DataFrame, title: str):
    """Bar chart of total forecast by item."""
    fig = px.bar(
        df,
        x="item",
        y="total_forecast",
        title=title,
        labels={"total_forecast": "Total Forecast", "item": "Item ID"},
        template="plotly_dark",
        color="total_forecast",
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white"
    )
    return fig

# 
# PAGE: OVERVIEW
# 
if page == " Overview":
    st.title(" Retail Demand Forecasting Dashboard")
    st.markdown("""
    This dashboard is powered by a **LightGBM** model trained on the
    *Store Item Demand Forecasting* dataset (Kaggle).

    **Dataset:** 5 years of daily sales across 10 stores and 50 items (500 series).

    **Model Performance (Validation Set):**
    """)

    # Performance table
    perf = {
        "Model":   ["LightGBM", "CatBoost", "XGBoost", "Random Forest", "Linear Regression"],
        "MAE":     [0.4665,     0.5610,     0.5879,    0.7668,          0.0],
        "RMSE":    [0.6368,     0.7240,     0.7745,    1.1638,          0.0],
        "R²":      [0.9995,     0.9994,     0.9993,    0.9983,          1.0],
    }
    perf_df = pd.DataFrame(perf)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model",  "LightGBM")
    col2.metric("MAE",         "0.47")
    col3.metric("RMSE",        "0.64")
    col4.metric("R²",          "0.9995")

    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.divider()

    # R² bar chart
    fig = px.bar(
        perf_df[perf_df["Model"] != "Linear Regression"],
        x="Model", y="R²",
        color="R²",
        color_continuous_scale="Blues",
        title="Model R² Comparison",
        template="plotly_dark"
    )
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("""
    ### How to Use This Dashboard
    | Page | What it does |
    |---|---|
    |  Single Prediction | Predict sales for one store, item, and date |
    |  Date Range Forecast | Forecast an entire date range with charts |
    |  Store Dashboard | All-item forecast summary for a store |
    |  CSV Upload | Upload a CSV and download predictions |
    """)

# 
# PAGE: SINGLE PREDICTION
# 
elif page == " Single Prediction":
    st.title(" Single Day Prediction")
    st.markdown("Predict sales for a specific **store**, **item**, and **date**.")

    col1, col2, col3 = st.columns(3)
    with col1:
        store = st.selectbox("Store ID", options=list(range(1, 11)), index=0)
    with col2:
        item  = st.selectbox("Item ID",  options=list(range(1, 51)), index=0)
    with col3:
        date  = st.date_input(
            "Date",
            value=datetime(2018, 1, 1),
            min_value=datetime(2013, 1, 1),
            max_value=datetime(2025, 12, 31)
        )

    if st.button(" Predict", type="primary", use_container_width=True):
        if not is_healthy:
            st.error("API is offline. Please start the FastAPI server.")
        else:
            with st.spinner("Calling API..."):
                try:
                    result = predict_single(store, item, date)
                    pred   = result["predicted_sales"]

                    st.success(f" Prediction complete!")

                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Store",           store)
                    col_b.metric("Item",            item)
                    col_c.metric("Date",            str(date))
                    col_d.metric("Predicted Sales", pred, delta="units")

                    # Fun gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pred,
                        title={"text": "Predicted Sales", "font": {"color": "white"}},
                        gauge={
                            "axis":  {"range": [0, 200], "tickcolor": "white"},
                            "bar":   {"color": "#00d4ff"},
                            "bgcolor": "#1e1e2e",
                            "steps": [
                                {"range": [0,   50],  "color": "#2d2d3d"},
                                {"range": [50,  100], "color": "#3d3d4d"},
                                {"range": [100, 200], "color": "#4d4d5d"},
                            ],
                            "threshold": {
                                "line":  {"color": "#ff6b6b", "width": 4},
                                "value": pred
                            }
                        },
                        number={"font": {"color": "white"}}
                    ))
                    fig.update_layout(
                        paper_bgcolor="#0e1117",
                        font_color="white",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.json(result)

                except Exception as e:
                    st.error(f"Error: {e}")

# 
# PAGE: DATE RANGE FORECAST
# 
elif page == " Date Range Forecast":
    st.title(" Date Range Forecast")
    st.markdown("Forecast daily sales for a store-item pair over a **date range**.")

    col1, col2 = st.columns(2)
    with col1:
        store = st.selectbox("Store ID", options=list(range(1, 11)), index=0)
        item  = st.selectbox("Item ID",  options=list(range(1, 51)), index=4)
    with col2:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2018, 1, 1),
            min_value=datetime(2013, 1, 1),
            max_value=datetime(2025, 12, 31)
        )
        end_date = st.date_input(
            "End Date",
            value=datetime(2018, 3, 31),
            min_value=datetime(2013, 1, 1),
            max_value=datetime(2025, 12, 31)
        )

    if st.button(" Forecast", type="primary", use_container_width=True):
        if not is_healthy:
            st.error("API is offline.")
        elif start_date > end_date:
            st.error("Start date must be before end date.")
        elif (end_date - start_date).days > 365:
            st.error("Date range cannot exceed 365 days.")
        else:
            with st.spinner("Generating forecast..."):
                try:
                    result = predict_range(store, item, start_date, end_date)

                    # Summary metrics
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Total Days",           result["total_days"])
                    col_b.metric("Total Predicted Sales", result["total_predicted_sales"])
                    col_c.metric("Avg Daily Sales",
                                 round(result["total_predicted_sales"] / result["total_days"], 1))

                    # Build DataFrame for plotting
                    fc_df = pd.DataFrame(result["forecasts"])
                    fc_df["date"] = pd.to_datetime(fc_df["date"])

                    # Line chart
                    st.plotly_chart(
                        plot_forecast(
                            fc_df,
                            f"Daily Sales Forecast — Store {store}, Item {item}"
                        ),
                        use_container_width=True
                    )

                    # Weekly aggregation
                    fc_df["week"] = fc_df["date"].dt.to_period("W").astype(str)
                    weekly = fc_df.groupby("week")["predicted_sales"].sum().reset_index()
                    weekly.columns = ["Week", "Weekly Sales"]

                    st.subheader("Weekly Forecast Summary")
                    col_left, col_right = st.columns(2)
                    with col_left:
                        st.dataframe(weekly, use_container_width=True, hide_index=True)
                    with col_right:
                        fig_w = px.bar(
                            weekly, x="Week", y="Weekly Sales",
                            title="Weekly Sales Forecast",
                            template="plotly_dark",
                            color="Weekly Sales",
                            color_continuous_scale="Blues"
                        )
                        fig_w.update_layout(
                            plot_bgcolor="#0e1117",
                            paper_bgcolor="#0e1117",
                            font_color="white",
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig_w, use_container_width=True)

                    # Raw data table (collapsible)
                    with st.expander(" View Raw Forecast Data"):
                        st.dataframe(fc_df, use_container_width=True, hide_index=True)

                        # Download button
                        csv = fc_df.to_csv(index=False)
                        st.download_button(
                            label=" Download Forecast CSV",
                            data=csv,
                            file_name=f"forecast_store{store}_item{item}.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"Error: {e}")

# 
# PAGE: STORE DASHBOARD
# 
elif page == " Store Dashboard":
    st.title(" Store Forecast Dashboard")
    st.markdown(
        "View a forecast summary for **all 50 items** in a store "
        "over the next N days."
    )

    col1, col2 = st.columns(2)
    with col1:
        store = st.selectbox("Select Store", options=list(range(1, 11)))
    with col2:
        days = st.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30, step=7)

    if st.button(" Generate Store Dashboard", type="primary", use_container_width=True):
        if not is_healthy:
            st.error("API is offline.")
        else:
            with st.spinner(f"Forecasting all items for Store {store} over {days} days..."):
                try:
                    result = get_store_summary(store, days)

                    # Top-level metrics
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Store", f"Store {store}")
                    col_b.markdown("### Forecast Period")
                    col_b.info(f"{result['forecast_start']} → {result['forecast_end']}")

                    col_c.metric(
                        "Total Store Forecast",
                        f"{result['total_store_forecast']:,} units"
                    )

                    st.metric("Total Items", result["total_items"])

                    # Item summary table
                    item_df = pd.DataFrame(result["item_forecasts"])
                    item_df.columns = ["Item ID", "Total Forecast", "Avg Daily Sales"]

                    col_left, col_right = st.columns([1, 2])
                    with col_left:
                        st.subheader(" Item Forecast Table")
                        st.dataframe(
                            item_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Total Forecast":  st.column_config.NumberColumn(format="%d units"),
                                "Avg Daily Sales": st.column_config.NumberColumn(format="%.1f"),
                            }
                        )

                    with col_right:
                        st.subheader(" Total Forecast by Item")
                        st.plotly_chart(
                            plot_bar_items(
                                item_df.rename(columns={
                                    "Item ID": "item",
                                    "Total Forecast": "total_forecast"
                                }),
                                f"Store {store} — Total Forecast by Item ({days} days)"
                            ),
                            use_container_width=True
                        )

                    # Top 10 / Bottom 10
                    st.divider()
                    col_top, col_bot = st.columns(2)

                    with col_top:
                        st.subheader("🔝 Top 10 Items by Forecast")
                        top10 = item_df.nlargest(10, "Total Forecast")
                        fig_top = px.bar(
                            top10, x="Item ID", y="Total Forecast",
                            title="Top 10 Items", template="plotly_dark",
                            color="Total Forecast", color_continuous_scale="Greens"
                        )
                        fig_top.update_layout(
                            plot_bgcolor="#0e1117",
                            paper_bgcolor="#0e1117",
                            font_color="white"
                        )
                        st.plotly_chart(fig_top, use_container_width=True)

                    with col_bot:
                        st.subheader(" Bottom 10 Items by Forecast")
                        bot10 = item_df.nsmallest(10, "Total Forecast")
                        fig_bot = px.bar(
                            bot10, x="Item ID", y="Total Forecast",
                            title="Bottom 10 Items", template="plotly_dark",
                            color="Total Forecast", color_continuous_scale="Reds"
                        )
                        fig_bot.update_layout(
                            plot_bgcolor="#0e1117",
                            paper_bgcolor="#0e1117",
                            font_color="white"
                        )
                        st.plotly_chart(fig_bot, use_container_width=True)

                    # Download
                    csv = item_df.to_csv(index=False)
                    st.download_button(
                        label=" Download Store Forecast CSV",
                        data=csv,
                        file_name=f"store{store}_forecast_{days}days.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Error: {e}")

# 
# PAGE: CSV UPLOAD
# 
elif page == " CSV Upload":
    st.title(" Batch Prediction via CSV Upload")
    st.markdown("""
    Upload a CSV file with columns **store**, **item**, **date**
    and download predictions for all rows.

    **Required CSV format:**
    ```
    store,item,date
    1,1,2018-01-01
    1,2,2018-01-01
    2,5,2018-02-14
    ```
    An optional `id` column is also supported (like Kaggle's test.csv).
    """)

    # Sample download
    sample_data = "store,item,date\n1,1,2018-01-01\n1,2,2018-01-01\n2,5,2018-02-14\n3,10,2018-03-15\n"
    st.download_button(
        label=" Download Sample CSV",
        data=sample_data,
        file_name="sample_input.csv",
        mime="text/csv"
    )

    st.divider()
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="Max 50,000 rows"
    )

    if uploaded_file is not None:
        # Preview the uploaded file
        try:
            preview_df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)   # reset file pointer after reading

            st.subheader(" Preview (first 10 rows)")
            st.dataframe(preview_df.head(10), use_container_width=True, hide_index=True)
            st.caption(f"Total rows: {len(preview_df):,}")

            # Check required columns
            required = {"store", "item", "date"}
            if not required.issubset(set(preview_df.columns)):
                st.error(f"CSV must have columns: {required}. Found: {set(preview_df.columns)}")
            else:
                if st.button(" Run Predictions", type="primary", use_container_width=True):
                    if not is_healthy:
                        st.error("API is offline.")
                    else:
                        with st.spinner(f"Predicting {len(preview_df):,} rows..."):
                            try:
                                result_csv_bytes = upload_csv_predict(
                                    uploaded_file.read(),
                                    uploaded_file.name
                                )

                                # Show results
                                result_df = pd.read_csv(io.BytesIO(result_csv_bytes))
                                st.success(f" Predictions complete for {len(result_df):,} rows!")

                                # Summary stats
                                col_a, col_b, col_c = st.columns(3)
                                col_a.metric("Total Rows",           len(result_df))
                                col_b.metric("Total Predicted Sales", int(result_df["predicted_sales"].sum()))
                                col_c.metric("Avg Predicted Sales",   round(result_df["predicted_sales"].mean(), 1))

                                # Preview
                                st.subheader(" Prediction Results (first 20 rows)")
                                st.dataframe(
                                    result_df.head(20),
                                    use_container_width=True,
                                    hide_index=True
                                )

                                # Distribution chart
                                fig = px.histogram(
                                    result_df, x="predicted_sales",
                                    nbins=50, title="Distribution of Predicted Sales",
                                    template="plotly_dark",
                                    color_discrete_sequence=["#00d4ff"]
                                )
                                fig.update_layout(
                                    plot_bgcolor="#0e1117",
                                    paper_bgcolor="#0e1117",
                                    font_color="white"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Download
                                st.download_button(
                                    label="Download Predictions CSV",
                                    data=result_csv_bytes,
                                    file_name="predictions.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )

                            except Exception as e:
                                st.error(f"Error during prediction: {e}")

        except Exception as e:
            st.error(f"Could not read CSV file: {e}")

# 
# FOOTER
# 
st.divider()
st.caption(
    "Built by **Nabin Katwal** · "
    "Retail Demand Forecasting Portfolio Project · "
    "LightGBM + FastAPI + Streamlit"
)
