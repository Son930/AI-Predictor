import json
import signal

import torch
from fastapi import FastAPI, HTTPException,Request
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import yfinance as yf
from fastapi.middleware.cors import CORSMiddleware
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from feature_engineering import (
    add_moving_averages, compute_rsi, categorize_rsi, compute_macd,
    generate_macd_signals, add_volatility, add_atr, add_relative_volume,
    add_volume_change, add_daily_range, add_price_change, add_day_of_week, add_month_of_year
)

app = FastAPI()

server_is_healthy = True

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8250",
    "http://127.0.0.1:8250",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TIINGO_API_KEY = "861e7708bad94378c6b680a8fc481253bab5db5a"
BASE_URL = "https://api.tiingo.com/"


class StockDataset(Dataset):
    def __init__(self, df, lookback=30):
        self.df = df
        self.lookback = lookback
        self.features = df.drop(columns=['close', 'trade_signal'], errors='ignore').values
        self.target = df['close'].values
        self.scaler = StandardScaler()

        # Ensure no NaN values in features
        self.features = np.nan_to_num(self.scaler.fit_transform(self.features))
        self.input_size = self.features.shape[1]

    def __len__(self):
        return len(self.df) - self.lookback

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.lookback]
        y = self.target[idx + self.lookback]

        # Replace NaN values with zeros (as a last resort)
        x = np.nan_to_num(x)
        y = np.nan_to_num(y)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TemporalFusionTransformer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x) # Removed hn, cn as they are not used
        output = self.fc(lstm_out[:, -1, :])
        return output



def shutdown_handler(signum, frame):
    global server_is_healthy
    server_is_healthy = False
    print("Server is shutting down...")


signal.signal(signal.SIGTERM, shutdown_handler)

@app.get("/stock/{ticker}")
def fetch_tiingo_data(ticker: str, data_type: str, years: int, return_df: bool = False):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    if data_type in ("price", "history"):
        url = f"{BASE_URL}iex/{ticker}/prices?startDate={start_date.strftime('%Y-%m-%d')}&endDate={end_date.strftime('%Y-%m-%d')}&token={TIINGO_API_KEY}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data)

            if df.empty:
                return None if return_df else []  # Return None for backend, empty list for frontend

            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            required_cols = ['close', 'high', 'low', 'open']
            if 'volume' in df.columns:
                required_cols.append('volume')

            for col in required_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            if return_df:
                return df  # Return DataFrame for backend
            return df.to_dict(orient="records")  # Return list for frontend

        except requests.exceptions.RequestException as e:
            print(f"Tiingo API Error: {e}")
            return None if return_df else []
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Error parsing JSON from Tiingo: {e}")
            return None if return_df else []

    return None if return_df else []



def generate_trading_signals(df):
    if df is None or df.empty:
        return df

    print("Inside generate_trading_signals")
    print(f"DF Index type: {df.index.dtype}")

    try:
        df.index = pd.to_datetime(df.index, errors='coerce')
        print(f"DF Index type after conversion: {type(df.index)}")

        df['SMA'] = df['close'].rolling(window=20).mean()
        df['LMA'] = df['close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['close'], window=14)
        df['MACD'] = calculate_macd(df['close'])['MACD']
        df['Signal'] = calculate_macd(df['close'])['Signal']
        df['Volatility'] = df['close'].rolling(window=20).std()
        df['ATR'] = calculate_atr(df, window=14)

        if 'volume' in df.columns:
            df['Relative_Volume'] = df['volume'] / df['volume'].rolling(window=20).mean()
            df['Volume_Change'] = df['volume'].pct_change()

        df['Daily_Range'] = df['high'] - df['low']
        df['Price_Change'] = df['close'].pct_change()
        df['Day_of_Week'] = df.index.dayofweek
        df['Month_of_Year'] = df.index.month

        df['trade_signal'] = 'Hold'
        df.loc[df['SMA'] > df['LMA'], 'trade_signal'] = 'Buy'
        df.loc[df['SMA'] < df['LMA'], 'trade_signal'] = 'Sell'

        return df
    except Exception as e:
        print(f"Error in generate_trading_signals: {e}")
        raise


def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast_length=12, slow_length=26, signal_length=9):
    fast_EMA = prices.ewm(span=fast_length, adjust=False).mean()
    slow_EMA = prices.ewm(span=slow_length, adjust=False).mean()
    macd = fast_EMA - slow_EMA
    signal = macd.ewm(span=signal_length, adjust=False).mean()
    return pd.DataFrame({'MACD': macd, 'Signal': signal})


def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr


def train_tft_model(df, filename="tft_model.pth"):
    dataset = StockDataset(df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = TemporalFusionTransformer(dataset.input_size, hidden_size=64, output_size=1)  # Use dynamic input size
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    model_dir = Path(__file__).parent / "AImodelSave"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / filename
    torch.save({'model_state_dict': model.state_dict(), 'input_size': dataset.input_size},  str(model_path))  # Save input size

    return model


# @app.post("/train_model")
# async def train_model(ticker: str):
#     df = fetch_tiingo_data(ticker, data_type="price", years=3)
#     if df is None:
#         raise HTTPException(status_code=400, detail="Failed to fetch data from Tiingo.")
#
#     df = generate_trading_signals(df)
#     trained_model = train_tft_model(df)
#     return {"message": f"Model for {ticker} trained successfully!"}


@app.post("/train_model")
async def train_model(request: Request): # Use Request object
    try:
        body = await request.json() # Get the JSON body
        print(f"Received request body: {body}") # Print it

        ticker = body.get("ticker")  # Get the ticker (handle missing key)
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker is required.")
        df = fetch_tiingo_data(ticker, data_type="price", years=3)
        if df is None or not isinstance(df, list):  # Ensure it's a valid list
            raise HTTPException(status_code=400, detail="Failed to fetch data from Tiingo.")

        df = pd.DataFrame(df)  # Convert the list to a DataFrame

        if df.empty:
            raise HTTPException(status_code=400, detail="No data available for training.")

        df = generate_trading_signals(df)
        train_tft_model(df)
        return {"message": f"Model for {ticker} trained successfully!"}

    except json.JSONDecodeError: # Handle JSON error
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")

    except Exception as e: # Handle other errors
        print(f"Error in train_model: {e}")
        raise HTTPException(status_code=500, detail=f"Error in train_model: {e}")

# @app.get("/predict")
# async def predict(ticker: str):
#     try:
#         try:
#             # Load the Pre-trained Model
#             model_dir = Path(__file__).parent / "AImodelSave"
#             model_path = model_dir / "tft_model.pth"
#             checkpoint = torch.load(str(model_path))
#             input_size = checkpoint['input_size']
#             model = TemporalFusionTransformer(input_size, hidden_size=64, output_size=1)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             model.eval()
#             print(f"✅ Model loaded successfully from: {model_path}")
#         except Exception as e:
#             print(f"❌ Error loading model: {e}")
#             raise HTTPException(status_code=500, detail="Model loading error")
#
#         # 1. Fetch Recent Data (using the ticker)
#         raw_data = fetch_tiingo_data(ticker, data_type="price", years=1)
#
#         # Convert list to DataFrame if necessary
#         if isinstance(raw_data, list):
#             df = pd.DataFrame(raw_data)
#         else:
#             df = raw_data  # Assume it's already a DataFrame
#
#         if df is None or df.empty:
#             raise HTTPException(status_code=400, detail=f"No data found for {ticker}")
#
#         # 2. Calculate Features (same as during training)
#         df = generate_trading_signals(df)
#
#         # 3. Prepare Input for Model
#         dataset = StockDataset(df)
#         dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#
#         predictions = []
#         for x_batch, _ in dataloader:
#             with torch.no_grad():
#                 raw_prediction = model(x_batch).item()
#                 print(f"🔍 Raw Prediction: {raw_prediction}")  # Debug the raw model output
#                 if not (raw_prediction == float('inf') or raw_prediction == float(
#                         '-inf') or raw_prediction != raw_prediction):  # NaN check
#                     predictions.append(raw_prediction)
#                 else:
#                     predictions.append(None)
#
#         return {"prediction": predictions[-1] if predictions else None}
#
#     except Exception as e:
#         print(f"❌ Prediction error: {e}")
#         raise HTTPException(status_code=500, detail=f"Prediction error: {e}")



@app.get("/predict")
async def predict(ticker: str):
    try:
        # Load Pre-trained Model
        model_dir = Path(__file__).parent / "AImodelSave"
        model_path = model_dir / "tft_model.pth"
        checkpoint = torch.load(str(model_path))
        input_size = checkpoint['input_size']
        model = TemporalFusionTransformer(input_size, hidden_size=64, output_size=1)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 1. Fetch Recent Data
        raw_data = fetch_tiingo_data(ticker, data_type="price", years=1)
        df = pd.DataFrame(raw_data) if isinstance(raw_data, list) else raw_data

        if df is None or df.empty:
            raise HTTPException(status_code=400, detail=f"No data found for {ticker}")

        # 2. Calculate Features
        df = generate_trading_signals(df)

        # 3. Prepare Input for Model
        dataset = StockDataset(df)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        predictions = []
        for x_batch, _ in dataloader:
            with torch.no_grad():
                raw_prediction = model(x_batch).item()
                predictions.append(raw_prediction)

        predicted_price = predictions[-1] if predictions else None

        # 4. Generate Buy/Sell/Hold Signal
        latest_row = df.iloc[-1]
        sma = latest_row["SMA"]
        lma = latest_row["LMA"]
        rsi = latest_row["RSI"]

        # Simple Rule-Based Decision
        if predicted_price is not None:
            if sma > lma and rsi < 70:
                trade_signal = "Buy"
            elif sma < lma and rsi > 30:
                trade_signal = "Sell"
            else:
                trade_signal = "Hold"
        else:
            trade_signal = "Hold"

        return {
            "predicted_price": predicted_price,
            "trade_signal": trade_signal
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/health")
def health_check():
    if server_is_healthy:
        return {"status": "healthy"}
    else:
        return {"status": "unhealthy"}


@app.post("/sent-anal")
def sentimental_anal():
    url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=demo'
    r = requests.get(url)
    data = r.json()

    return data