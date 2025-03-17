import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# Load the trained LSTM model
model = keras.models.load_model("stock_model.keras")
print("Model Input Shape:", model.input_shape)  # Debugging


def get_stock_data(stock_symbol, start_date, end_date):
    """Fetches stock data with Open, High, Low, Close, Volume."""
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]  # Ensure 5 features


def prepare_data(stock_data):
    """Scales the stock data and returns scaler + scaled data."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)
    return scaler, scaled_data


def predict_future_prices(model, stock_data, start_year, end_year):
    """Predicts future stock prices using LSTM."""
    scaler, scaled_data = prepare_data(stock_data)

    last_60_days = scaled_data[-60:, :]  # Select last 60 days of ALL features
    future_predictions = []
    current_input = last_60_days.reshape(1, 60, 5)  # Reshaped to match model input

    num_months = (end_year - start_year + 1) * 12
    date_range = pd.date_range(start=f"{start_year}-01-01", periods=num_months, freq='M')

    for i in range(num_months):
        predicted_price = model.predict(current_input)[0][0]  # Predict only Close price
        future_predictions.append(predicted_price)

        # Append predicted Close price, keep other features same (zero placeholder)
        new_entry = np.append(current_input[:, 1:, :], [[[predicted_price] + [0] * 4]], axis=1)
        current_input = new_entry  # Shift the window

    # 🔹 Fix: Create a dummy array with 5 columns for inverse_transform
    dummy_array = np.zeros((num_months, 5))
    dummy_array[:, 3] = future_predictions  # Assign predicted Close prices to the correct column

    # Apply inverse transform
    future_prices = scaler.inverse_transform(dummy_array)[:, 3]  # Extract only Close prices

    return date_range, future_prices.flatten()


def main():
    st.title("Stock Price Prediction (Multi-Year Forecast)")
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., TCS.BO, RELIANCE.NS)", "TCS.BO")
    start_year = st.number_input("Enter Start Year", min_value=2024, max_value=2035, value=2024)
    end_year = st.number_input("Enter End Year", min_value=start_year, max_value=2035, value=2026)

    if st.button("Predict"):
        stock_data = get_stock_data(stock_symbol, "2014-01-01", "2024-12-31")

        if len(stock_data) < 60:
            st.error("Not enough historical data (60 days required).")
            return

        date_range, future_prices = predict_future_prices(model, stock_data, start_year, end_year)

        # Plot the results
        fig, ax = plt.subplots()
        ax.plot(date_range, future_prices, label="Predicted Prices", linestyle='dashed', color='r')
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.set_title(f"Stock Price Prediction for {stock_symbol} ({start_year}-{end_year})")
        ax.legend()
        st.pyplot(fig)

        # Display prediction results
        df = pd.DataFrame({"Date": date_range, "Predicted Price": future_prices})
        st.write(df)


if __name__ == "__main__":
    main()
