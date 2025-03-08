import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

# Get stock symbol from user
stock_symbol = input("Enter BSE stock symbol (e.g., RELIANCE.BO): ")

# Fetch last 10 years of stock data
start_date = (datetime.today() - timedelta(days=3650)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

df = yf.download(stock_symbol, start=start_date, end=end_date)

# Plot stock prices
plt.figure(figsize=(12,5))
plt.plot(df['Close'], label="Closing Price")
plt.title(f"{stock_symbol} Stock Price Over 10 Years")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Prepare data for LSTM
df_close = df[['Close']]
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df_close)

# Create sequences for training
def create_sequences(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60  # Using past 60 days to predict next day
X, Y = create_sequences(df_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshaping for LSTM

# Split data into train & test sets (80% training, 20% testing)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Build LSTM Model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test))

# Plot Training Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Predict Next 2 Years
future_days = 730  # 2 years
future_inputs = df_scaled[-time_step:].reshape(1, time_step, 1)

future_preds = []
for _ in range(future_days):
    pred = model.predict(future_inputs)[0][0]
    future_preds.append(pred)
    future_inputs = np.append(future_inputs[:, 1:, :], [[[pred]]], axis=1)

# Convert Predictions Back to Original Scale
future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# Create Future Dates
future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, future_days+1)]

# Plot Future Predictions
plt.figure(figsize=(12,5))
plt.plot(df.index, df['Close'], label="Historical Prices")
plt.plot(future_dates, future_preds, label="Predicted Prices (Next 2 Years)", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"Future Stock Price Prediction for {stock_symbol}")
plt.legend()
plt.show()

# Save the trained model
model.save("bse_stock_lstm_model.h5")
print("Model saved successfully!")