import random

import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
# Streamlit UI
st.title("📈 Stock Price Prediction (Next 6 Months)")

# List of selected stocks (replace with your BSE stock tickers)
stock_list = ["RELIANCE.BO", "TCS.BO", "INFY.BO", "HDFCBANK.BO", "ICICIBANK.BO", "SBIN.BO", "AXISBANK.BO",
              "BAJFINANCE.BO", "HINDUNILVR.BO", "KOTAKBANK.BO", "ITC.BO", "MARUTI.BO", "LT.BO", "ASIANPAINT.BO",
              "SUNPHARMA.BO", "NESTLEIND.BO", "TITAN.BO", "ULTRACEMCO.BO", "WIPRO.BO", "HCLTECH.BO"]

selected_stocks = st.multiselect("Select Stocks", stock_list, default=stock_list[:5])

# LSTM Parameters
lookback = 60  # Use last 60 days for prediction
future_days = 180  # Predict next 6 months

# Store results
predictions = {}

def fetch_stock_data(ticker):
    """ Fetch historical data for the given stock. """
    stock = yf.download(ticker, period="10y", interval="1d")
    return stock['Close'].dropna()

def prepare_data(data):
    """ Prepare dataset for LSTM model. """
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(data_scaled) - future_days):
        X.append(data_scaled[i - lookback:i, 0])
        y.append(data_scaled[i + future_days, 0])

    return np.array(X), np.array(y), scaler

def build_lstm_model():
    """ Build and compile the LSTM model. """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_stock(stock):
    """ Train LSTM model and predict next 6 months price. """
    st.write(f"🔄 Processing {stock}...")
    data = fetch_stock_data(stock)
    if len(data) < lookback + future_days:
        st.warning(f"⚠️ Skipping {stock} due to insufficient data.")
        return

    X, y, scaler = prepare_data(data)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train LSTM model
    model = build_lstm_model()
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Predict next price
    last_60_days = data[-lookback:].values.reshape(-1, 1)
    last_60_days_scaled = scaler.transform(last_60_days)
    X_future = np.array([last_60_days_scaled]).reshape(1, lookback, 1)

    predicted_price_scaled = model.predict(X_future)[0, 0]
    predicted_price = scaler.inverse_transform([[predicted_price_scaled]])[0, 0]

    last_price = data.iloc[-1]
    percentage_change = ((predicted_price - last_price) / last_price) * 100

    predictions[stock] = {
        "Current Price": last_price,
        "Predicted Price (6M)": predicted_price,
        "Percentage Change": percentage_change
    }

# Process selected stocks
for stock in selected_stocks:
    predict_stock(stock)

# Display predictions in a table with color coding
if predictions:
    pred_df = pd.DataFrame(predictions).T
    pred_df = pred_df.reset_index().rename(columns={"index": "Stock"})

    # Apply color styling
    def highlight_change(val):
        """ Highlight percentage change with green (positive) and red (negative). """
        if isinstance(val, (int, float, np.number)):  # Ensure val is scalar
            color = 'green' if val > 0 else 'red'
            return f'background-color: {color}; color: white'
        return ''

    st.write("### 📊 Prediction Results")
    st.dataframe(pred_df.style.applymap(highlight_change, subset=["Percentage Change"]))
