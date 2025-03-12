import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set Stock Symbol & Download Data
stock_symbol = "RELIANCE.BO"  # Change to any BSE stock symbol
df = yf.download(stock_symbol, period="10y")

# Check if data is empty
if df.empty:
    print("No stock data found! Check the stock symbol.")
    exit()

# Select Close Price & Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
df_close = df[['Close']].values
df_scaled = scaler.fit_transform(df_close)

# Save the scaler for future use
np.save("scaler.npy", scaler)

# Prepare Data for LSTM
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60  # Number of previous days used for prediction
X, y = create_sequences(df_scaled, time_step)

# Reshape for LSTM (Samples, Time Steps, Features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split Data into Training & Testing
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1)
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

# Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Save Trained Model
model.save("stock_model.keras")
print("✅ Model training completed and saved as stock_model.keras")

# Plot Training Loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("LSTM Training Loss")
plt.show()
