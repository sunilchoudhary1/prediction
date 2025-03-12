import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
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

# Fetch Stock Data
@st.cache_data
def fetch_stock_data(symbol):
    df = yf.download(symbol, period="10y")
    df.reset_index(inplace=True)
    return df

df = fetch_stock_data(stock_symbol)

# Stop if no data is found
if df.empty:
    st.error("❌ No data found. Try another stock.")
    st.stop()

# Convert Date Column
df['Date'] = pd.to_datetime(df['Date'])
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

# Select Month for Display
months = [date.strftime('%B %Y') for date in future_dates]
selected_month = st.sidebar.selectbox("📅 Select a month for prediction:", months)
selected_index = months.index(selected_month)
selected_price = future_preds[selected_index][0]

# ✅ Bar Chart (Replaces Line Chart)
fig, ax = plt.subplots(figsize=(12, 6))
sns.set_style("whitegrid")

# Plot Historical Prices as Bars
ax.bar(df['Date'][-50:], df['Close'][-50:], label="Historical Prices", color="#0174DF", alpha=0.7)

# Plot Predicted Prices as Bars
ax.bar(pred_df['Date'], pred_df['Predicted Close'], label="Predicted Prices", color="#DF0101", alpha=0.7)

# Highlight Selected Prediction as a Separate Bar
ax.bar(pred_df['Date'].iloc[selected_index], selected_price, color="black", label=f"Prediction for {selected_month}")

# Customize Graph Appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Closing Price", fontsize=12)
ax.set_title(stock_symbol, fontsize=16)
ax.legend()

# ✅ Display Bar Chart in Streamlit
st.pyplot(fig)

# ✅ Show Predicted Price in UI
st.write(f"### 📌 Predicted Price for {selected_month}: ₹{selected_price:.2f}")
st.success("✅ Prediction Completed!")
