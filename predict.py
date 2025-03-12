import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load Model & Scaler
model = load_model("stock_model.keras")
scaler = np.load("scaler.npy", allow_pickle=True).item()

# Streamlit UI Configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction App")
st.sidebar.header("🔍 Select Stock & Settings")

# User Input: Stock Symbol
stock_symbol = st.sidebar.text_input("Enter BSE stock symbol (e.g., RELIANCE.BO):", "RELIANCE.BO")


# Fetch Stock Data (Cache for Performance)
@st.cache_data
def fetch_stock_data(symbol):
    df = yf.download(symbol, period="10y")
    df.reset_index(inplace=True)
    return df


df = fetch_stock_data(stock_symbol)

# Stop if No Data Found
if df.empty:
    st.error("❌ No data found. Try another stock.")
    st.stop()

# Convert Date Column
df['Date'] = pd.to_datetime(df['Date'])

# Moving Average for Smoother Trend Analysis
df['MA50'] = df['Close'].rolling(window=50).mean()

# Prepare Close Prices for Prediction
df_close = df[['Close']].values
df_scaled = scaler.transform(df_close)

# User Input: Future Months
future_months = st.sidebar.slider("📅 Predict Future Months:", 1, 24, 12)

# Prediction Process
time_step = 60
future_inputs = df_scaled[-time_step:].reshape(1, time_step, 1)
future_preds, future_dates = [], []
last_date = df['Date'].iloc[-1]

for i in range(future_months):
    pred = model.predict(future_inputs, verbose=0)[0][0]
    future_preds.append(pred)
    future_inputs = np.append(future_inputs[:, 1:, :], [[[pred]]], axis=1)
    next_month = last_date + timedelta(days=30 * (i + 1))
    future_dates.append(next_month)

# Convert Predictions to Original Scale
future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# Create Prediction DataFrame
pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds.flatten()})

# Fill the gap between last known value and future predictions
gap_filler_dates = pd.date_range(start=last_date, end=future_dates[0], periods=10)[1:]
gap_filler_values = np.linspace(df['Close'].iloc[-1], future_preds[0][0], num=len(gap_filler_dates))
gap_df = pd.DataFrame({'Date': gap_filler_dates, 'Predicted Close': gap_filler_values.flatten()})
pred_df = pd.concat([gap_df, pred_df], ignore_index=True)

# Select Month for Display
months = [date.strftime('%B %Y') for date in pred_df['Date']]
selected_month = st.sidebar.selectbox("📅 Select a month for prediction:", months)
selected_index = months.index(selected_month)
selected_price = float(pred_df['Predicted Close'].iloc[selected_index])

# Generate Headline Based on Prediction Trend
price_change = float(selected_price) - float(df['Close'].iloc[-1])

if price_change > 0:
    headline = f"🚀 **Stock Price Surge Expected!** {stock_symbol} is projected to rise by ₹{price_change:.2f}."
elif price_change < 0:
    headline = f"📉 **Stock Price Drop Alert!** {stock_symbol} might decline by ₹{abs(price_change):.2f}."
else:
    headline = f"📊 **Stable Market Ahead!** No major change expected for {stock_symbol}."

# ✅ Enhanced Graph Visualization
theme_color = "#1E90FF"  # Blue Theme
fig = go.Figure()

# Candlestick Chart for Historical Prices
fig.add_trace(go.Candlestick(
    x=df['Date'].tail(100),
    open=df['Open'].tail(100),
    high=df['High'].tail(100),
    low=df['Low'].tail(100),
    close=df['Close'].tail(100),
    name="Historical Prices",
    increasing_line_color='#32CD32',
    decreasing_line_color='#FF4500',
    opacity=0.7
))

# Moving Average (MA50)
fig.add_trace(go.Scatter(
    x=df['Date'].tail(100),
    y=df['MA50'].tail(100),
    mode='lines',
    name='50-Day MA',
    line=dict(color='#FFD700', width=2, dash='dot')
))

# Predicted Prices with Gap Filling
fig.add_trace(go.Scatter(
    x=pred_df['Date'],
    y=pred_df['Predicted Close'],
    mode='lines+markers',
    name='Predicted Prices',
    line=dict(color=theme_color, width=2, dash='solid'),
    marker=dict(color=theme_color, size=6, symbol="circle-open")
))

# Highlight Selected Month Prediction with Annotation
fig.add_trace(go.Scatter(
    x=[pred_df['Date'].iloc[selected_index]],
    y=[selected_price],
    mode='markers+text',
    marker=dict(color='black', size=12, symbol="star"),
    text=[f'₹{selected_price:.2f}'],
    textposition="top center",
    name=f'Prediction for {selected_month}'
))

# ✅ Chart Customization
fig.update_layout(
    title=f"📊 {stock_symbol} Stock Price Prediction",
    xaxis_title="Date",
    yaxis_title="Stock Price",
    template="plotly_white",
    plot_bgcolor='rgba(240,240,240,0.9)',
    paper_bgcolor='white',
    xaxis_rangeslider_visible=False,
    height=700,
    width=1200,
    font=dict(family="Arial", size=14),
    hovermode="x unified",
    xaxis=dict(showgrid=True, gridcolor='LightGray'),
    yaxis=dict(showgrid=True, gridcolor='LightGray')
)

# ✅ Display Graph in Streamlit
st.plotly_chart(fig, use_container_width=True)

# ✅ Show Prediction Headline
st.markdown(f"## {headline}")

# ✅ Show Predicted Price in UI
st.markdown(f"### 📌 Predicted Price for **{selected_month}**: ₹{selected_price:.2f}")

st.success("✅ Prediction Completed! 🎯")