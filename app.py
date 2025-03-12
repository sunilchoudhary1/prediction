import streamlit as st
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Load Trained Model and Scaler
model = load_model("stock_model.keras")  # Ensure this file exists
scaler = np.load("scaler.npy", allow_pickle=True).item()  # Load the saved scaler

# Streamlit UI
st.title("📈 Stock Price Prediction App")
st.sidebar.header("🔍 Select Stock & Settings")

# User Input: Stock Symbol
stock_symbol = st.sidebar.text_input("Enter BSE stock symbol (e.g., RELIANCE.BO):", "RELIANCE.BO")

# Fetch Stock Data (Cached for Performance)
@st.cache_data
def fetch_stock_data(symbol):
    df = yf.download(symbol, period="10y")
    return df

df = fetch_stock_data(stock_symbol)

# Error Handling for Empty Data
if df.empty:
    st.error("❌ No data found for the given stock symbol.")
    st.stop()

# Preprocess Stock Data
df_close = df[['Close']].values
df_scaled = scaler.transform(df_close)

# User Input: Number of Future Months
future_months = st.sidebar.slider("📅 Predict Future Months:", min_value=1, max_value=24, value=12)

# Prediction
time_step = 60
future_inputs = df_scaled[-time_step:].reshape(1, time_step, 1)
future_preds = []
future_dates = []

for i in range(future_months):
    pred = model.predict(future_inputs, verbose=0)[0][0]
    future_preds.append(pred)
    future_inputs = np.append(future_inputs[:, 1:, :], [[[pred]]], axis=1)
    next_month = df.index[-1] + timedelta(days=(i + 1) * 30)
    future_dates.append(next_month)

# Convert Predictions Back to Original Scale
future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# Plot Graph
fig = go.Figure()

# Historical Prices
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Prices', line=dict(color='blue')))

# Predicted Prices (Dotted Line)
fig.add_trace(go.Scatter(x=future_dates, y=future_preds.flatten(), mode='lines+markers',
                         name='Predicted Prices', line=dict(dash='dot', color='red')))

# Update Layout (✅ Fixed Parenthesis Issue)
fig.update_layout(
    title=f"📊 Stock Price Prediction for {stock_symbol}",
    xaxis_title="Date",
    yaxis_title="Stock Price",
    legend_title="Legend"
)

# Display Plot
st.plotly_chart(fig)
st.success("✅ Prediction Completed!")
