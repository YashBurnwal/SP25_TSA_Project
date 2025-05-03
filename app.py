import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
import time
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ✅ Metric evaluator
def evaluate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

# App setup
st.set_page_config(page_title="Stock Forecasting App", layout="wide")
st.title("📈 Stock Forecasting App")

# Constants
train_start = "2023-01-01"
train_end = "2025-01-01"
future_start = "2025-01-02"
future_end = "2025-02-05"
forecast_steps = 30

# Session state
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False
if "data" not in st.session_state:
    st.session_state.data = None
if "future_data" not in st.session_state:
    st.session_state.future_data = None
if "models_output" not in st.session_state:
    st.session_state.models_output = {}

# Step 1 – Stock selection
stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "JNJ", "WMT"]
col1, _ = st.columns([1, 5])  # Left 1 part, right 5 parts (invisible)
with col1:
    ticker = st.selectbox("Step 1: Select Stock Ticker", stocks)

if st.button("✅ Load Data"):
    data = yf.download(ticker, start=train_start, end=train_end)
    future_data = yf.download(ticker, start=future_start, end=future_end)
    if not data.empty and not future_data.empty:
        st.session_state.data = data
        st.session_state.future_data = future_data
        st.session_state.data_loaded = True
        time.sleep(5)
        st.success("Data loaded successfully.")
    else:
        st.error("Failed to load data.")

# Step 2 – Train models
if st.session_state.data_loaded:
    st.markdown("---")
    if st.button("🚀 Train Models"):
        data = st.session_state.data
        future_data = st.session_state.future_data
        close = data['Close'].copy()
        close = close.asfreq('B')
        close.fillna(method='ffill', inplace=True)
        results = {}

        progress = st.progress(0)
        status = st.empty()

        # Train SARIMA
        status.text("Training: SARIMA...")
        sarima = SARIMAX(close, order=(1,1,1), seasonal_order=(1,1,1,30)).fit(disp=False)
        sarima_pred = sarima.get_forecast(steps=forecast_steps).predicted_mean[:len(future_data)]
        actual_sarima = future_data['Close'].values[:len(sarima_pred)]
        results['SARIMA'] = (sarima_pred.values,
                             *evaluate_metrics(actual_sarima, sarima_pred.values))
        progress.progress(33)
        time.sleep(10)
        # Train Holt-Winters
        status.text("Training: Holt-Winters...")
        es_model = ExponentialSmoothing(close, trend='add', seasonal='add', seasonal_periods=30).fit()
        es_pred = es_model.forecast(steps=forecast_steps)[:len(future_data)]
        actual_es = future_data['Close'].values[:len(es_pred)]
        results['Holt-Winters'] = (es_pred.values,
                                   *evaluate_metrics(actual_es, es_pred.values))
        progress.progress(66)
        time.sleep(10)
        # Train LSTM
        status.text("Training: LSTM...")
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close.values.reshape(-1, 1))
        seq_len = 20
        X, y = [], []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i-seq_len:i])
            y.append(scaled[i])
        X, y = np.array(X), np.array(y)
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        last_seq = scaled[-seq_len:]
        fut_preds = []
        for _ in range(forecast_steps):
            input_seq = last_seq[-seq_len:].reshape(1, seq_len, 1)
            next_scaled = model.predict(input_seq)[0][0]
            fut_preds.append(next_scaled)
            last_seq = np.append(last_seq, [[next_scaled]], axis=0)

        fut_preds_inv = scaler.inverse_transform(np.array(fut_preds).reshape(-1, 1)).flatten()
        actual_vals = future_data['Close'].values
        pred_vals = fut_preds_inv[:len(actual_vals)]
        results['LSTM'] = (fut_preds_inv, *evaluate_metrics(actual_vals, pred_vals))

        progress.progress(100)
        time.sleep(5)
        status.text("✅ Training Complete.")
        st.session_state.models_output = results
        st.session_state.models_trained = True
        st.success("Models trained and evaluated.")

# Step 3 – Forecast visualization
if st.session_state.models_trained:
    st.markdown("---")
    models_output = st.session_state.models_output
    future_data = st.session_state.future_data

    # Metrics table
    metrics_df = pd.DataFrame(columns=["Model", "MAE", "RMSE", "MAPE (%)"])
    for model_name, (_, mae, rmse, mape) in models_output.items():
        new_row = pd.DataFrame([{
            "Model": model_name,
            "MAE": f"{mae:.2f}",
            "RMSE": f"{rmse:.2f}",
            "MAPE (%)": f"{mape:.2f}"
        }])
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    col1, col2 = st.columns([2, 3])  # Adjust widths as needed
    with col1:
        selected_model = st.selectbox("Step 3: Select a model", metrics_df["Model"])
    with col2:
        forecast_days = st.selectbox("Forecast Horizon (days)", [5, 10, 15, 30])

    st.dataframe(metrics_df.set_index("Model"))

    if st.button("📊 Show Forecast"):
        forecast = models_output[selected_model][0][:forecast_days]
        actual = future_data['Close'].values[:forecast_days]
        forecast_dates = future_data.index[:forecast_days]

        fig, ax = plt.subplots(figsize=(6, 3))  # Compact size

        # Plotting
        ax.plot(forecast_dates, actual, label='Actual', linewidth=2)
        ax.plot(forecast_dates, forecast, label=f'{selected_model} Forecast', linestyle='--', linewidth=2)

        # Labels & title (smaller fonts)
        ax.set_title(f'{selected_model} Forecast vs Actual for {ticker} ({forecast_days} Days)', fontsize=6)
        ax.set_xlabel("Date", fontsize=7)
        ax.set_ylabel("Price (USD)", fontsize=7)

        # Ticks
        ax.tick_params(axis='x', labelsize=4)
        ax.tick_params(axis='y', labelsize=4)

        # Legend
        ax.legend(fontsize=4)
        ax.grid(True)
        fig.autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig)
        
    # === PREDICT FROM TODAY BUTTON ===
    if st.button("🔮 Predict Future from Today"):
        today = pd.Timestamp.today().date()
        hist_start = "2023-01-01"
        hist_data = yf.download(ticker, start=hist_start, end=today)
        close = hist_data['Close'].copy().asfreq('B')
        close.fillna(method='ffill', inplace=True)

        future_dates = pd.bdate_range(start=today + pd.Timedelta(days=1), periods=forecast_days)

        if selected_model == "SARIMA":
            model = SARIMAX(close, order=(1,1,1), seasonal_order=(1,1,1,30)).fit(disp=False)
            forecast = model.get_forecast(steps=forecast_days).predicted_mean
        elif selected_model == "Holt-Winters":
            model = ExponentialSmoothing(close, trend='add', seasonal='add', seasonal_periods=30).fit()
            forecast = model.forecast(steps=forecast_days)
        elif selected_model == "LSTM":
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(close.values.reshape(-1, 1))
            seq_len = 20
            X = []
            for i in range(seq_len, len(scaled)):
                X.append(scaled[i-seq_len:i])
            X = np.array(X)
            X_train = X.reshape((X.shape[0], X.shape[1], 1))

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, scaled[seq_len:], epochs=10, batch_size=32, verbose=0)

            last_seq = scaled[-seq_len:]
            preds = []
            for _ in range(forecast_days):
                input_seq = last_seq[-seq_len:].reshape(1, seq_len, 1)
                next_scaled = model.predict(input_seq)[0][0]
                preds.append(next_scaled)
                last_seq = np.append(last_seq, [[next_scaled]], axis=0)
            forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

        # === PLOT FUTURE PREDICTION ===
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(future_dates, forecast, label=f'{selected_model} Prediction', linestyle='--', linewidth=2)
        ax.set_title(f'{selected_model} Forecast from Today ({forecast_days} Days)', fontsize=6)
        ax.set_xlabel("Date", fontsize=7)
        ax.set_ylabel("Predicted Price (USD)", fontsize=7)
        ax.tick_params(axis='x', labelsize=4)
        ax.tick_params(axis='y', labelsize=4)
        ax.legend(fontsize=4)
        ax.grid(True)
        fig.autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig)
