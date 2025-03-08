from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained LSTM model
model = load_model("bse_stock_lstm_model.h5")  # Make sure this file exists


def fetch_stock_data(symbol):
    start_date = (datetime.today() - timedelta(days=3650)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(symbol, start=start_date, end=end_date)
    return df


def prepare_data(df):
    df_close = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df_close)

    time_step = 60
    X = df_scaled[-time_step:].reshape(1, time_step, 1)

    future_preds = []
    for _ in range(730):  # Predict for 2 years
        pred = model.predict(X)[0][0]
        future_preds.append(pred)
        X = np.append(X[:, 1:, :], [[[pred]]], axis=1)

    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 731)]

    return df, future_dates, future_preds


def plot_stock_data(df, future_dates, future_preds):
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['Close'], label="Historical Prices")
    plt.plot(future_dates, future_preds, label="Predicted Prices (Next 2 Years)", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Prediction")
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


@app.route('/predict', methods=['GET'])
def predict_stock():
    stock_symbol = request.args.get('symbol', '').replace('BSE:', '') + '.BO'

    try:
        df = fetch_stock_data(stock_symbol)
        df, future_dates, future_preds = prepare_data(df)
        img_base64 = plot_stock_data(df, future_dates, future_preds)

        return jsonify({
            "symbol": stock_symbol,
            "prediction_chart": f"data:image/png;base64,{img_base64}"
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
