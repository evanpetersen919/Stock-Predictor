# ----------------------------------- Libraries -----------------------------------
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import requests
# ---------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------
# Loads historical stock data and engineers features for model training.
# ticker: Stock ticker symbol (e.g. "AAPL", "MSFT", "TSLA", "^GSPC")
# period: Data period to download (default "max" for maximum available data)
def load_and_engineer_data(ticker, period="max"):
    df = yf.Ticker(ticker).history(period=period)
    if 'Dividends' in df.columns:
        del df['Dividends']
    if 'Stock Splits' in df.columns:
        del df['Stock Splits']
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    df = df.loc["1990-01-01":].copy()
    horizons = [2, 5, 60, 250, 1000]
    new_predictors = []
    for horizon in horizons:
        rolling_averages = df.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        df[ratio_column] = df["Close"] / rolling_averages["Close"]
        trend_column = f"Trend_{horizon}"
        df[trend_column] = df.shift(1).rolling(horizon).sum()["Target"]
        new_predictors += [ratio_column, trend_column]
    df = df.dropna()
    predictors = ["Close", "Volume", "Open", "High", "Low"] + new_predictors
    return df, predictors
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# Trains a Random Forest model on the provided data and predictors.
# df: DataFrame with historical stock data and engineered features
# predictors: List of predictor column names
def train_model(df, predictors):
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1, class_weight='balanced')
    model.fit(df[predictors], df["Target"])
    return model
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# Predicts the probability and direction (Up/Down) for the next trading day.
# model: Trained RandomForestClassifier
# latest_data: DataFrame with the latest stock data (single row)
# predictors: List of predictor column names
def predict_next_day(model, latest_data, predictors):
    if latest_data.isnull().any().any():
        latest_data = latest_data.fillna(method='ffill', axis=0).fillna(method='bfill', axis=0)
    prob_up = model.predict_proba(latest_data[predictors])[:, 1][0]
    prediction = "Up" if prob_up >= 0.5 else "Down"
    return prediction, prob_up
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# Fetches historical stock data from the Azure web service.
# ticker: Stock ticker symbol (e.g. "AAPL", "MSFT", "TSLA", "^GSPC")
# Returns: DataFrame with historical stock data
def fetch_historical_data(ticker):
    url = f'https://stockpredictor-ffevaadwd7dygwba.westus3-01.azurewebsites.net/history?ticker={ticker}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        return df
    else:
        print(f"Error fetching data: {response.status_code}")
        return None
# ---------------------------------------------------------------------------------