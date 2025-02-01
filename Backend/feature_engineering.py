import numpy as np
import pandas as pd

def add_moving_averages(df, short_window=20, long_window=100):
    """Add Simple Moving Averages (SMA)."""
    df['SMA'] = df['close'].rolling(window=short_window).mean().bfill()
    df['LMA'] = df['close'].rolling(window=long_window).mean().bfill()

    return df

def compute_rsi(df, window=14):
    """Compute the Relative Strength Index (RSI)."""
    delta = df["close"].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Ensure RSI has the same length as df and fill NaN with 50
    return rsi.fillna(50)  # Default to 50 if RSI cannot be computed

def categorize_rsi(df, rsi_column='RSI'):
    """Categorize RSI values as Overbought, Oversold, or Neutral."""
    conditions = [
        (df[rsi_column] < 30),
        (df[rsi_column] > 70),
        (df[rsi_column] >= 30) & (df[rsi_column] <= 70)
    ]
    choices = ['Oversold', 'Overbought', 'Neutral']
    df['RSI_category'] = np.select(conditions, choices, default='Neutral')
    return df

def compute_macd(df, short=12, long=26, signal=9):
    """Calculate MACD and Signal line, ensuring no NaN issues."""
    macd = df['close'].ewm(span=short, adjust=False).mean() - df['close'].ewm(span=long, adjust=False).mean()
    signal_line = macd.ewm(span=signal, adjust=False).mean()

    # Ensure MACD and Signal have the same length as df and fill NaN with 0
    return macd.fillna(0), signal_line.fillna(0)

def generate_macd_signals(df):
    """Generate Buy or Sell signals based on MACD and Signal line crossovers."""
    df['trade_signal'] = "0"  # Default value for signal

    # Generate Buy signal when MACD crosses above Signal
    df.loc[(df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1)), 'trade_signal'] = "1"

    # Generate Sell signal when MACD crosses below Signal
    df.loc[(df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1)), 'trade_signal'] = "-1"

    # Fill any remaining NaN trade signals with "0"
    df['trade_signal'] = df['trade_signal'].fillna("0")

    return df

def add_volatility(df, window=20):
    df['Volatility'] = df['close'].rolling(window=window).std().bfill()
    return df

def add_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = np.maximum.reduce([high_low, high_close, low_close], 0)
    df['ATR'] = true_range.rolling(window=window).mean().bfill()
    return df

def add_relative_volume(df, window=20):
    df['Relative_Volume'] = df['volume'] / df['volume'].rolling(window).mean().bfill()
    return df

def add_volume_change(df):
    df['Volume_Change'] = df['volume'].pct_change().bfill()
    return df

def add_daily_range(df):
    df['Daily_Range'] = df['high'] - df['low']
    return df

def add_price_change(df):
    df['Price_Change'] = df['close'].pct_change().bfill()
    return df

def add_day_of_week(df):
    df['Day_of_Week'] = df.index.dayofweek
    return df

def add_month_of_year(df):
    df['Month_of_Year'] = df.index.month
    return df


