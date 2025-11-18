
import numpy as np
import pandas as pd
import argparse
from scipy.signal import find_peaks, butter, filtfilt

# Optional deps (kept behind flags for speed)
try:
    import pywt  # only used if include_slow=True
except Exception:
    pywt = None

# -----------------------------
# Helpers
# -----------------------------
def _require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def _std_volume_name(df):
    # Map either 'tick_volume' or 'volume' to a standard column 'vol'
    if 'tick_volume' in df.columns:
        df = df.rename(columns={'tick_volume': 'vol'})
    elif 'volume' in df.columns:
        df = df.rename(columns={'volume': 'vol'})
    else:
        raise ValueError("Input must contain 'tick_volume' or 'volume' column.")
    return df

def _encode_rsi_category(series):
    mapping = {'Oversold': -1, 'Neutral': 0, 'Overbought': 1}
    return series.map(mapping).astype('int8', errors='ignore')

# -----------------------------
# Core, fast features (vectorized)
# -----------------------------
def add_moving_averages(df, short_window=20, long_window=100):
    df['SMA'] = df['close'].rolling(short_window, min_periods=short_window).mean()
    df['LMA'] = df['close'].rolling(long_window, min_periods=long_window).mean()
    return df

def compute_rsi(df, window=14):
    delta = df['close'].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    # Category (string -> later encoded)
    conds = [df['RSI'] < 30, df['RSI'] > 70]
    choices = ['Oversold', 'Overbought']
    df['RSI_category'] = np.select(conds, choices, default='Neutral')
    return df

def compute_macd(df, short=12, long=26, signal=9):
    ema_short = df['close'].ewm(span=short, adjust=False).mean()
    ema_long = df['close'].ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    df['MACD'] = macd
    df['Signal'] = signal_line
    # cross signals
    cross_up = (df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1))
    cross_dn = (df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1))
    df['trade_signal'] = np.where(cross_up, 1, np.where(cross_dn, -1, 0)).astype('int8')
    return df

def add_volatility(df, window=20):
    df['Volatility'] = df['close'].rolling(window, min_periods=window).std()
    return df

def add_atr(df, window=14):
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window, min_periods=window).mean()
    return df

def add_volume_features(df, window=20):
    df['Relative_Volume'] = df['vol'] / df['vol'].rolling(window, min_periods=window).mean()
    df['Volume_Change'] = df['vol'].pct_change()
    return df

def add_price_features(df):
    df['Daily_Range'] = df['high'] - df['low']
    df['Price_Change'] = df['close'].pct_change()
    return df

def add_calendar(df):
    if isinstance(df.index, pd.DatetimeIndex):
        df['Day_of_Week'] = df.index.dayofweek.astype('int8')
        df['Month_of_Year'] = df.index.month.astype('int8')
        # hour/session if intraday
        try:
            df['Hour_of_Day'] = df.index.hour.astype('int8')
            df['Is_Asian_Session'] = ((df['Hour_of_Day'] >= 0) & (df['Hour_of_Day'] < 8)).astype('int8')
            df['Is_European_Session'] = ((df['Hour_of_Day'] >= 8) & (df['Hour_of_Day'] < 16)).astype('int8')
            df['Is_American_Session'] = ((df['Hour_of_Day'] >= 16) & (df['Hour_of_Day'] < 24)).astype('int8')
            df['Day_of_Month'] = df.index.day.astype('int8')
            df['Is_Month_Start'] = (df['Day_of_Month'] <= 5).astype('int8')
            df['Is_Month_End'] = (df['Day_of_Month'] >= 25).astype('int8')
        except Exception:
            pass
    return df

def add_wavelet_features(df, wavelet='db4', level=2):
    if pywt is None:
        return df
    prices = df['close'].values
    coeffs = pywt.wavedec(prices, wavelet, level=level)
    # Denoise: keep approximation only
    denoised_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    denoised = pywt.waverec(denoised_coeffs, wavelet)[:len(prices)]
    df['Denoised_Price'] = denoised
    # Pad detail coefficients
    df['Wavelet_D1'] = np.pad(coeffs[-1], (0, len(prices) - len(coeffs[-1])), mode='constant')
    df['Wavelet_D4'] = np.pad(coeffs[-level], (0, len(prices) - len(coeffs[-level])), mode='constant')
    return df

def add_lagged_prices(df, lags=(1, 7, 14)):
    for lag in lags:
        df[f'Price_t-{lag}'] = df['close'].shift(lag)
    return df

def add_bollinger_bands(df, window=20, k=2):
    sma = df['close'].rolling(window, min_periods=window).mean()
    std = df['close'].rolling(window, min_periods=window).std()
    df['BB_Upper'] = sma + k * std
    df['BB_Lower'] = sma - k * std
    df['BB_Bandwidth'] = (df['BB_Upper'] - df['BB_Lower']) / sma
    return df

def add_ema(df, short_span=10, long_span=50):
    df['EMA_Short'] = df['close'].ewm(span=short_span, adjust=False).mean()
    df['EMA_Long'] = df['close'].ewm(span=long_span, adjust=False).mean()
    return df

def add_roc(df, period=14):
    df['ROC'] = df['close'].pct_change(periods=period) * 100.0
    return df

def add_stochastic_oscillator(df, period=14, smooth=3):
    low_n = df['low'].rolling(period, min_periods=period).min()
    high_n = df['high'].rolling(period, min_periods=period).max()
    denom = (high_n - low_n).replace(0, np.nan)
    stoch_k = (df['close'] - low_n) / denom * 100.0
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_k.rolling(smooth, min_periods=smooth).mean()
    return df

def add_lagged_returns(df, lags=(1, 7, 14)):
    for lag in lags:
        df[f'Return_t-{lag}'] = df['close'].pct_change(periods=lag)
    return df

def add_price_action_patterns(df):
    # Bearish engulfing example
    prev_up = df['open'].shift(1) < df['close'].shift(1)
    curr_dn = df['open'] > df['close']
    engulf = (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))
    df['engulfing'] = (prev_up & curr_dn & engulf).astype('int8')
    return df

def add_cycle_indicator(df, window=20):
    mean = df['close'].rolling(window, min_periods=window).mean()
    std = df['close'].rolling(window, min_periods=window).std()
    df['price_cycle'] = (df['close'] - mean) / std
    return df

def add_mean_reversion_indicator(df, window=20):
    ma = df['close'].rolling(window, min_periods=window).mean()
    df['distance_from_ma'] = (df['close'] - ma) / ma
    return df

def add_momentum_features(df, lookback=10):
    df['momentum'] = df['close'] - df['close'].shift(lookback)
    return df

def add_peak_trough_features(df, prominence=0.01):
    prices = df['close'].values
    scale = np.nanmean(prices)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    peak_idx, _ = find_peaks(prices, prominence=prominence * scale)
    trough_idx, _ = find_peaks(-prices, prominence=prominence * scale)
    is_peak = np.zeros(len(df), dtype=np.int8); is_peak[peak_idx] = 1
    is_trough = np.zeros(len(df), dtype=np.int8); is_trough[trough_idx] = 1
    df['Is_Peak'] = is_peak
    df['Is_Trough'] = is_trough
    # distance since last peak/trough (vectorized)
    peak_pos = np.where(is_peak == 1)[0]
    trough_pos = np.where(is_trough == 1)[0]
    def dist_since_last(idx_array, n):
        dist = np.full(n, np.nan, dtype=float)
        if idx_array.size == 0:
            return dist
        last = -np.inf
        j = 0
        for i in range(n):
            if j < len(idx_array) and i == idx_array[j]:
                last = i; j += 1
            dist[i] = np.nan if last == -np.inf else i - last
        return dist
    n = len(df)
    df['Dist_Last_Peak'] = dist_since_last(peak_pos, n)
    df['Dist_Last_Trough'] = dist_since_last(trough_pos, n)
    return df

def add_filtered_signals(df):
    # Low-pass Butterworth filter (fast; O(n))
    b, a = butter(N=2, Wn=0.1, btype='low')
    prices = df['close'].values
    if np.all(np.isfinite(prices)) and len(prices) > 10:
        filtered = filtfilt(b, a, prices, method='gust')
        df['Filtered_Price'] = filtered
        df['Filter_Ratio'] = df['close'] / np.where(filtered == 0, np.nan, filtered)
    else:
        df['Filtered_Price'] = np.nan
        df['Filter_Ratio'] = np.nan
    return df

def add_adaptive_indicators(df):
    vol = df['close'].rolling(20, min_periods=20).std()
    atr = df['ATR']
    vol_mean100 = vol.rolling(100, min_periods=100).mean()
    scale = (vol / vol_mean100).replace([np.inf, -np.inf], np.nan)
    k = 2.5
    df['Adaptive_Upper'] = df['close'] + (atr * k * scale)
    df['Adaptive_Lower'] = df['close'] - (atr * k * scale)
    df['Adaptive_Position'] = (df['close'] - df['Adaptive_Lower']) / (df['Adaptive_Upper'] - df['Adaptive_Lower'])
    return df

def add_keltner_channels(df, window=20, multiplier=2):
    typical = (df['high'] + df['low'] + df['close']) / 3.0
    ema = typical.ewm(span=window, adjust=False).mean()
    atr = df['ATR']
    df['Keltner_Middle'] = ema
    df['Keltner_Upper'] = ema + multiplier * atr
    df['Keltner_Lower'] = ema - multiplier * atr
    df['Keltner_Position'] = (df['close'] - df['Keltner_Lower']) / (df['Keltner_Upper'] - df['Keltner_Lower'])
    df['Keltner_Width'] = (df['Keltner_Upper'] - df['Keltner_Lower']) / ema
    return df

def add_fibonacci_retracements(df, window=100):
    rolling_high = df['high'].rolling(window, min_periods=window).max()
    rolling_low = df['low'].rolling(window, min_periods=window).min()
    rng = rolling_high - rolling_low
    df['Fib_0'] = rolling_low
    df['Fib_23_6'] = rolling_low + 0.236 * rng
    df['Fib_38_2'] = rolling_low + 0.382 * rng
    df['Fib_50_0'] = rolling_low + 0.5 * rng
    df['Fib_61_8'] = rolling_low + 0.618 * rng
    df['Fib_100'] = rolling_high

    tol = 0.01 * rng
    for level in ['0','23_6','38_2','50_0','61_8','100']:
        lvl_col = f'Fib_{level}'
        near_col = f'Near_{lvl_col}'
        df[near_col] = ((df['close'] >= df[lvl_col] - tol) & (df['close'] <= df[lvl_col] + tol)).astype('int8')
    return df

def add_ichimoku_cloud(df):
    nine_high = df['high'].rolling(9, min_periods=9).max()
    nine_low = df['low'].rolling(9, min_periods=9).min()
    df['Conversion_Line'] = (nine_high + nine_low) / 2.0

    high26 = df['high'].rolling(26, min_periods=26).max()
    low26 = df['low'].rolling(26, min_periods=26).min()
    df['Base_Line'] = (high26 + low26) / 2.0

    df['Leading_Span_A'] = ((df['Conversion_Line'] + df['Base_Line']) / 2.0).shift(26)

    high52 = df['high'].rolling(52, min_periods=52).max()
    low52 = df['low'].rolling(52, min_periods=52).min()
    df['Leading_Span_B'] = ((high52 + low52) / 2.0).shift(26)

    df['Lagging_Span'] = df['close'].shift(-26)
    df['Cloud_Strength'] = df['Leading_Span_A'] - df['Leading_Span_B']
    df['Above_Cloud'] = ((df['close'] > df['Leading_Span_A']) & (df['close'] > df['Leading_Span_B'])).astype('int8')
    df['Below_Cloud'] = ((df['close'] < df['Leading_Span_A']) & (df['close'] < df['Leading_Span_B'])).astype('int8')
    df['In_Cloud'] = (~((df['Above_Cloud'] == 1) | (df['Below_Cloud'] == 1))).astype('int8')
    return df

def add_zscore_features(df, window=20):
    roll_mean = df['close'].rolling(window, min_periods=window).mean()
    roll_std = df['close'].rolling(window, min_periods=window).std()
    df['Price_ZScore'] = (df['close'] - roll_mean) / roll_std

    vol_mean = df['vol'].rolling(window, min_periods=window).mean()
    vol_std = df['vol'].rolling(window, min_periods=window).std()
    df['Volume_ZScore'] = (df['vol'] - vol_mean) / vol_std

    rsi_mean = df['RSI'].rolling(window, min_periods=window).mean()
    rsi_std = df['RSI'].rolling(window, min_periods=window).std()
    df['RSI_ZScore'] = (df['RSI'] - rsi_mean) / rsi_std

    volat = df['Volatility']
    vmean = volat.rolling(window, min_periods=window).mean()
    vstd = volat.rolling(window, min_periods=window).std()
    df['Volatility_ZScore'] = (volat - vmean) / vstd

    df['Extreme_Price'] = (df['Price_ZScore'].abs() > 2).astype('int8')
    df['Extreme_Volume'] = (df['Volume_ZScore'].abs() > 2).astype('int8')
    return df

def add_obv_mfi_chaikin(df):
    # OBV
    obv = np.zeros(len(df), dtype=float)
    close = df['close'].values
    vol = df['vol'].values
    up = close[1:] > close[:-1]
    dn = close[1:] < close[:-1]
    obv[1:][up] = vol[1:][up]
    obv[1:][dn] = -vol[1:][dn]
    df['OBV'] = np.cumsum(obv)
    # MFI (14)
    typical = (df['high'] + df['low'] + df['close']) / 3.0
    money_flow = typical * df['vol']
    pos_flow = np.where(typical > typical.shift(1), money_flow, 0.0)
    neg_flow = np.where(typical < typical.shift(1), money_flow, 0.0)
    pos_sum = pd.Series(pos_flow, index=df.index).rolling(14, min_periods=14).sum()
    neg_sum = pd.Series(neg_flow, index=df.index).rolling(14, min_periods=14).sum()
    mr = pos_sum / (neg_sum.replace(0, np.nan))
    df['MFI'] = 100 - (100 / (1 + mr))
    # Chaikin Money Flow (20)
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
    mfv = mfm * df['vol']
    df['Chaikin_MF'] = mfv.rolling(20, min_periods=20).sum() / df['vol'].rolling(20, min_periods=20).sum()
    return df

# -----------------------------
# Targets
# -----------------------------
def add_target(df, horizon=5, threshold=0.0002):
    future = df['close'].shift(-horizon)
    future_return = (future - df['close'])  # absolute difference
    df['Future_Return'] = future_return
    # classify (-1, 0, 1) by absolute change relative to threshold
    df['Target'] = np.where(future_return > threshold, 1, np.where(future_return < -threshold, -1, 0)).astype('int8')
    return df

# -----------------------------
# Finalization
# -----------------------------
def _finalize(df, drop_leaky=True, encode_cats=True, drop_na=True):
    if encode_cats and 'RSI_category' in df.columns:
        df['RSI_category'] = _encode_rsi_category(df['RSI_category'])
    if drop_leaky and 'Future_Return' in df.columns:
        df = df.drop(columns=['Future_Return'])
    # No backward fills! Avoid look-ahead leakage.
    if drop_na:
        df = df.dropna()
    else:
        df = df.fillna(0)
    return df

# -----------------------------
# Main pipeline
# -----------------------------
def engineer_features(df, fast=True, lookback=None, include_slow=False, return_Xy=False,
                      target_horizon=5, target_threshold=0.0002, drop_na=True):
    """
    Fast, RF-ready feature set.

    Params:
      fast: prefer fast features only (True by default)
      lookback: only compute on the last N rows (recommended for live)
      include_slow: include slow/expensive features (wavelets, filter, etc.)
      return_Xy: return (X, y) instead of full df
      target_horizon: steps ahead for target
      target_threshold: classification threshold for +/-1
      drop_na: drop rows with NaNs (prevents leakage)
    """
    _require_cols(df, ['open', 'high', 'low', 'close'])
    df = _std_volume_name(df.copy())

    if lookback is not None and lookback > 0:
        df = df.tail(lookback).copy()

    # Core fast set
    add_moving_averages(df)
    compute_rsi(df)
    compute_macd(df)
    add_volatility(df)
    add_atr(df)
    add_volume_features(df)
    add_price_features(df)
    add_calendar(df)
    add_lagged_prices(df)
    add_bollinger_bands(df)
    add_ema(df)
    add_roc(df)
    add_stochastic_oscillator(df)
    add_lagged_returns(df)
    add_price_action_patterns(df)
    add_cycle_indicator(df)
    add_mean_reversion_indicator(df)
    add_momentum_features(df)
    add_peak_trough_features(df)
    add_keltner_channels(df)
    add_fibonacci_retracements(df)
    add_ichimoku_cloud(df)
    add_zscore_features(df)
    add_obv_mfi_chaikin(df)
    add_target(df, horizon=target_horizon, threshold=target_threshold)

    # Optional slow features (off by default)
    if include_slow:
        add_filtered_signals(df)
        if pywt is not None:
            add_wavelet_features(df)

    # Finalize (no leakage!)
    df = _finalize(df, drop_leaky=True, encode_cats=True, drop_na=drop_na)

    if return_Xy:
        if 'Target' not in df.columns:
            raise ValueError("Target not found after engineering.")
        X = df.drop(columns=['Target'])
        y = df['Target'].astype(int).values
        return X, y
    return df

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Fast, RF-ready feature engineering for MT5 OHLCV data.")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV file")
    parser.add_argument('--output', type=str, required=True, help="Path to output CSV file")
    parser.add_argument('--lookback', type=int, default=0, help="Only compute on last N rows if > 0")
    parser.add_argument('--include_slow', action='store_true', help="Include slow/expensive features")
    parser.add_argument('--no_drop_na', action='store_true', help="Do not drop NaNs (fill with 0 instead)")
    parser.add_argument('--horizon', type=int, default=5, help="Target horizon (bars ahead)")
    parser.add_argument('--threshold', type=float, default=0.0002, help="Target threshold")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if 'timestamp' in df.columns:
        # Detect milliseconds vs seconds
        ts = pd.to_datetime(df['timestamp'], errors='coerce')
        if ts.isna().mean() > 0.9:  # probably ms
            ts = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df['timestamp'] = ts
        df = df.set_index('timestamp')
    else:
        raise ValueError("Input CSV must have a 'timestamp' column.")

    engineered = engineer_features(
        df,
        fast=True,
        lookback=args.lookback if args.lookback and args.lookback > 0 else None,
        include_slow=args.include_slow,
        return_Xy=False,
        target_horizon=args.horizon,
        target_threshold=args.threshold,
        drop_na=not args.no_drop_na
    )
    engineered.to_csv(args.output, index=True)
    print(f"Saved engineered features to {args.output}")

if __name__ == '__main__':
    main()
