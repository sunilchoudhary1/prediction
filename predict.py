import random

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)



# Streamlit UI Configuration
st.set_page_config(page_title="Stock Prediction", layout="wide")
st.title("📈 Stock Price Prediction")

st.sidebar.header("Stock Selection & Settings")
stock_symbol = st.sidebar.text_input("Enter BSE stock symbol (e.g., RELIANCE.BO):", "RELIANCE.BO")

# Function to Fetch Stock Data
@st.cache_data
def fetch_stock_data(symbol):
    try:
        df = yf.download(symbol, period="10y")  # Try fetching 10 years of data
        if df is None or df.empty:  # If no data, try maximum available data
            df = yf.download(symbol, period="max")
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

df = fetch_stock_data(stock_symbol)
if df.empty:
    st.error("❌ No data found. Try another stock.")
    st.stop()

df['Date'] = pd.to_datetime(df['Date'])

# Compute RSI Indicator
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = compute_rsi(df['Close'])
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()

# Normalize Data
scaler = MinMaxScaler(feature_range=(0, 1))
feature_columns = ['Close', 'Volume', 'RSI', 'SMA_50', 'SMA_200']
df.fillna(method='bfill', inplace=True)
data_scaled = scaler.fit_transform(df[feature_columns])

# Prepare LSTM Data
def prepare_lstm_data(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = prepare_lstm_data(data_scaled, time_step)
X = X.reshape(X.shape[0], time_step, X.shape[2])

# Load or Train LSTM Model
try:
    model = load_model("stock_model.keras")
except:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_step, X.shape[2])),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=30, batch_size=16, verbose=1)
    model.save("stock_model.keras")

# Predict Future Prices
future_months = st.sidebar.slider("📅 Predict Future Months:", 1, 24, 24)
future_inputs = data_scaled[-time_step:].copy()
future_preds = []
future_dates = []
last_date = df['Date'].iloc[-1]

for i in range(future_months):
    prediction_input = future_inputs.reshape(1, time_step, X.shape[2])
    pred = model.predict(prediction_input, verbose=0)[0][0]
    future_preds.append(pred)
    new_feature_set = np.roll(future_inputs, -1, axis=0)
    new_feature_set[-1, 0] = pred
    future_inputs = new_feature_set
    future_dates.append(last_date + timedelta(days=30 * (i + 1)))

# Convert Predictions Back to Original Scale
future_preds = scaler.inverse_transform(
    np.concatenate((np.array(future_preds).reshape(-1, 1), np.zeros((future_months, X.shape[2] - 1))), axis=1))[:, 0]
pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds})

# Sidebar: Select Prediction Period
st.sidebar.subheader("📅 Select Prediction Period")
months_list = [date.strftime("%B %Y") for date in pred_df["Date"]]

start_month = st.sidebar.selectbox("From:", months_list, index=0)
end_month = st.sidebar.selectbox("To:", months_list, index=len(months_list) - 1)

# Convert selected months to datetime for filtering
start_date = datetime.strptime(start_month, "%B %Y")
end_date = datetime.strptime(end_month, "%B %Y")

# Filter predictions based on selected period
filtered_pred_df = pred_df[(pred_df["Date"] >= start_date) & (pred_df["Date"] <= end_date)]

# Update Graph
# Determine Colors for Increasing/Decreasing Prices
filtered_pred_df["Color"] = np.where(filtered_pred_df["Predicted Close"].diff() > 0, "green", "red")
fig = go.Figure()
fig = go.Figure()

# Separate increasing and decreasing segments
for i in range(1, len(filtered_pred_df)):
    price_prev = filtered_pred_df["Predicted Close"].iloc[i - 1]
    price_curr = filtered_pred_df["Predicted Close"].iloc[i]
    color = "green" if price_curr > price_prev else "red"

    fig.add_trace(go.Scatter(
        x=[filtered_pred_df["Date"].iloc[i - 1], filtered_pred_df["Date"].iloc[i]],
        y=[price_prev, price_curr],
        mode="lines+markers",
        line=dict(color=color, width=3),
        marker=dict(size=8, color=color),
        showlegend=False
    ))

st.plotly_chart(fig, use_container_width=True)

# Calculate Price Change
price_start = filtered_pred_df.iloc[0]["Predicted Close"]
price_end = filtered_pred_df.iloc[-1]["Predicted Close"]
price_change = price_end - price_start
price_change_percent = (price_change / price_start) * 100

# Display Price Change Analysis
st.markdown(f"## 📉 Price Change Analysis")
st.markdown(f"- **Start Price ({start_month}):** ₹{price_start:.2f}")
st.markdown(f"- **End Price ({end_month}):** ₹{price_end:.2f}")
st.markdown(f"- **Change:** ₹{price_change:.2f} ({price_change_percent:.2f}%)")

# Indicate whether price increased or decreased
if price_change > 0:
    st.success(f"📈 Price increased by {price_change_percent:.2f}%")
else:
    st.error(f"📉 Price decreased by {price_change_percent:.2f}%")

