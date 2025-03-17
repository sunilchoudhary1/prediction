from turtle import st

import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


# Fetch Stock Data
@st.cache_data
def fetch_stock_data(symbol):
    try:
        df = yf.download(symbol, period="10y")  # Try fetching 10 years of data
        if df.empty:  # If no data, try maximum available data
            df = yf.download(symbol, period="max")
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure


# Compute RSI Indicator
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# Prepare Data for LSTM
def prepare_lstm_data(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


# Train and Save Model
def train_model(stock_symbol="WIPRO.BO", time_step=60):
    print(f"Fetching data for {stock_symbol}...")
    df = fetch_stock_data(stock_symbol)

    if df.empty:
        print("❌ No data found. Exiting...")
        return

    df['RSI'] = compute_rsi(df['Close'])
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    feature_columns = ['Close', 'Volume', 'RSI', 'SMA_50', 'SMA_200']
    df.fillna(method='bfill', inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df[feature_columns])

    X, y = prepare_lstm_data(data_scaled, time_step)
    X = X.reshape(X.shape[0], time_step, X.shape[2])

    print("Training LSTM Model...")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_step, X.shape[2])),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=50, batch_size=16, verbose=1)

    model.save("stock_model.keras")
    print("✅ Model saved successfully!")


if __name__ == "__main__":
    train_model()
