# http://localhost:8000/index.html
# python backend.py
# python -m http.server 8000

# -------------------- Libraries -----------------------------
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from model_utils import load_and_engineer_data, train_model, predict_next_day
# -------------------------------------------------------------


# -------------------- App Setup ------------------------------
app = Flask(__name__)
CORS(app)
# -------------------------------------------------------------


# -------------------- Routes / History -----------------------
@app.route('/history', methods=['POST'])
def history():
    data = request.get_json()
    ticker = data.get('ticker', '^GSPC')
    period = data.get('range', 'max')
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty:
            return jsonify({"error": "No data found for ticker."}), 404
        df = df.reset_index()
        chart_data = {
            "dates": df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            "close": df['Close'].round(2).tolist(),
            "period_high": round(df['High'].max(), 2),
            "period_low": round(df['Low'].min(), 2)
        }
        return jsonify(chart_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# -------------------------------------------------------------

# -------- Prediction Endpoint -------- -----------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get('ticker', '^GSPC')
    df, predictors = load_and_engineer_data(ticker, period="max")
    if len(df) < 2:
        return jsonify({"error": "Not enough data for prediction."}), 400
    train = df.iloc[:-1]
    test = df.iloc[[-1]]
    model = train_model(train, predictors)
    prediction, prob_up = predict_next_day(model, test, predictors)
    return jsonify({
        "prediction": prediction,
        "probability_up": round(prob_up*100, 2)
    })
# ---------------------------------------------------

# -------------------- Main -------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
# ---------------------------------------------------
