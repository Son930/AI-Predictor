import json
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
        self.features = df.drop(columns=['close', 'trade_signal'], errors='ignore').values # errors='ignore' handles missing columns in prediction
        self.target = df['close'].values
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        self.input_size = self.features.shape[1] # Store input size

    def __len__(self):
        return len(self.df) - self.lookback

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.lookback]
        y = self.target[idx + self.lookback]
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


def fetch_tiingo_data(ticker: str, data_type: str, years: int):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    if data_type in ("price", "history"):
        url = f"{BASE_URL}iex/{ticker}/prices?startDate={start_date.strftime('%Y-%m-%d')}&endDate={end_date.strftime('%Y-%m-%d')}&token={TIINGO_API_KEY}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            required_cols = ['close', 'high', 'low', 'open']
            if 'volume' in df.columns:
                required_cols.append('volume')

            if any(col in df.columns for col in required_cols):
                for col in required_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                df = df[[col for col in required_cols if col in df.columns]]
                return df
            else:
                print(f"Error: No required price data ('close', 'high', 'low', 'open', 'volume') found for {ticker}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Tiingo API Error: {e}")
            return None
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Error parsing JSON from Tiingo: {e}")
            return None
    else:
        return None


def generate_trading_signals(df):
    if df is None or df.empty:
        return df

    print("Inside generate_trading_signals")
    print(f"DF Index type: {df.index.dtype}")

    try:
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
        df.index = pd.to_datetime(df.index)
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
        if df is None:
            raise HTTPException(status_code=400, detail="Failed to fetch data from Tiingo.")

        df = generate_trading_signals(df)
        train_tft_model(df)
        return {"message": f"Model for {ticker} trained successfully!"}

    except json.JSONDecodeError: # Handle JSON error
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")

    except Exception as e: # Handle other errors
        print(f"Error in train_model: {e}")
        raise HTTPException(status_code=500, detail=f"Error in train_model: {e}")

@app.get("/predict")
async def predict(ticker: str):
    try:
        try:
            # Load the Pre-trained Model
            model_dir = Path(__file__).parent / "AImodelSave"
            model_path = model_dir / "tft_model.pth"
            checkpoint = torch.load(str(model_path))
            input_size = checkpoint['input_size']
            model = TemporalFusionTransformer(input_size, hidden_size=64, output_size=1)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"Model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail="Model loading error")

        # 1. Fetch Recent Data (using the ticker)
        df = fetch_tiingo_data(ticker, data_type="price", years=1)

        if df is None or df.empty:
            raise HTTPException(status_code=400, detail=f"No data found for {ticker}")

        # 2. Calculate Features (same as during training)
        df = generate_trading_signals(df)

        # 3. Prepare Input for Model
        dataset = StockDataset(df)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        predictions = []
        for x_batch, _ in dataloader:
            with torch.no_grad():
                raw_prediction = model(x_batch).item()
                print(f"Raw Prediction: {raw_prediction}")  # Debug the raw model output
                if not (raw_prediction == float('inf') or raw_prediction == float(
                        '-inf') or raw_prediction != raw_prediction):  # NaN check
                    predictions.append(raw_prediction)
                else:
                    predictions.append(None)

        return {"prediction": predictions[-1] if predictions else None}

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


print("end of program, thank you")