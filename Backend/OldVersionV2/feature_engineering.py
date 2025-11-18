import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import MinMaxScaler
import argparse
import scipy.stats as stats
from scipy.signal import find_peaks, butter, filtfilt
import talib  


def add_moving_averages(df, short_window=20, long_window=100):
    df['SMA'] = df['close'].rolling(window=short_window).mean().bfill()
    df['LMA'] = df['close'].rolling(window=long_window).mean().bfill()
    return df

def compute_rsi(df, window=14):
    delta = df["close"].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def categorize_rsi(df, rsi_column='RSI'):
    conditions = [(df[rsi_column] < 30), (df[rsi_column] > 70), (df[rsi_column] >= 30) & (df[rsi_column] <= 70)]
    choices = ['Oversold', 'Overbought', 'Neutral']
    df['RSI_category'] = np.select(conditions, choices, default='Neutral')
    return df

def compute_macd(df, short=12, long=26, signal=9):
    macd = df['close'].ewm(span=short, adjust=False).mean() - df['close'].ewm(span=long, adjust=False).mean()
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd.fillna(0), signal_line.fillna(0)

def generate_macd_signals(df):
    df['trade_signal'] = "0"
    df.loc[(df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1)), 'trade_signal'] = "1"
    df.loc[(df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1)), 'trade_signal'] = "-1"
    df['trade_signal'] = df['trade_signal'].fillna("0")
    return df

def add_volatility(df, window=20):
    df['Volatility'] = df['close'].rolling(window=window).std().bfill()
    return df

def add_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.Series(np.maximum.reduce([high_low, high_close, low_close]), index=df.index)
    df['ATR'] = true_range.rolling(window=window).mean().bfill()
    return df

def add_relative_volume(df, window=20):
    df['Relative_Volume'] = df['tick_volume'] / df['tick_volume'].rolling(window).mean().bfill()
    return df

def add_volume_change(df):
    df['Volume_Change'] = df['tick_volume'].pct_change().bfill()  # Use tick_volume as in MT5 data
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

def add_wavelet_features(df, wavelet='db4', level=2):
    prices = df['close'].values
    coeffs = pywt.wavedec(prices, wavelet, level=level)
    denoised_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    denoised = pywt.waverec(denoised_coeffs, wavelet)[:len(prices)]
    df['Denoised_Price'] = denoised
    df['Wavelet_D1'] = np.pad(coeffs[-1], (0, len(prices) - len(coeffs[-1])), mode='constant')
    df['Wavelet_D4'] = np.pad(coeffs[-level], (0, len(prices) - len(coeffs[-level])), mode='constant')
    return df

def add_lagged_prices(df, lags=[1, 7, 14]):
    for lag in lags:
        df[f'Price_t-{lag}'] = df['close'].shift(lag).bfill()
    return df

def add_bollinger_bands(df, window=20, k=2):
    sma = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    df['BB_Upper'] = sma + (k * std)
    df['BB_Lower'] = sma - (k * std)
    df['BB_Bandwidth'] = (df['BB_Upper'] - df['BB_Lower']) / sma
    df[['BB_Upper', 'BB_Lower', 'BB_Bandwidth']] = df[['BB_Upper', 'BB_Lower', 'BB_Bandwidth']].bfill()
    return df

def add_ema(df, short_span=10, long_span=50):
    df['EMA_Short'] = df['close'].ewm(span=short_span, adjust=False).mean()
    df['EMA_Long'] = df['close'].ewm(span=long_span, adjust=False).mean()
    return df

def add_roc(df, period=14):
    df['ROC'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
    df['ROC'] = df['ROC'].bfill()
    return df

def add_stochastic_oscillator(df, period=14, smooth=3):
    low_n = df['low'].rolling(window=period).min()
    high_n = df['high'].rolling(window=period).max()
    df['Stoch_K'] = (df['close'] - low_n) / (high_n - low_n) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(window=smooth).mean()
    df[['Stoch_K', 'Stoch_D']] = df[['Stoch_K', 'Stoch_D']].bfill()
    return df

def add_lagged_returns(df, lags=[1, 7, 14]):
    for lag in lags:
        df[f'Return_t-{lag}'] = (df['close'] - df['close'].shift(lag)) / df['close'].shift(lag)
    df[[f'Return_t-{lag}' for lag in lags]] = df[[f'Return_t-{lag}' for lag in lags]].bfill()
    return df

def normalize_features(df, exclude_cols=['trade_signal', 'RSI_category', 'Target']):
    scaler = MinMaxScaler()
    features_to_scale = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return df

def add_target(df):
    df['Future_Return'] = df['close'].shift(-5) - df['close']
    df['Target'] = np.where(df['Future_Return'] > 0.0002, 1, np.where(df['Future_Return'] < -0.0002, -1, 0))
    return df


def add_price_action_patterns(df):
    df['engulfing'] = ((df['open'].shift(1) < df['close'].shift(1)) & 
                        (df['open'] > df['close']) & 
                        (df['open'] > df['close'].shift(1)) & 
                        (df['close'] < df['open'].shift(1))).astype(int)
    return df

def add_cycle_indicator(df):
    df['price_cycle'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
    return df

def add_mean_reversion_indicator(df):
    df['distance_from_ma'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).mean()
    return df

def add_momentum_features(df):
    df['momentum'] = df['close'] - df['close'].shift(10)
    return df

# new 15 features: 
def add_fractal_dimension(df, window=20):
    """Calculate Fractal Dimension using Higuchi method."""
    def higuchi_fd(x, kmax=5):
        n = len(x)
        lk = np.zeros(kmax)
        for k in range(1, kmax + 1):
            lm = np.zeros(k)
            for m in range(k):
                ll = 0
                for i in range(1, int((n - m) / k)):
                    ll += abs(x[m + i * k] - x[m + (i - 1) * k])
                ll = ll * (n - 1) / (int((n - m) / k) * k) / k
                lm[m] = ll
            lk[k - 1] = np.mean(lm)
        
        x_data = np.log(np.arange(1, kmax + 1))
        y_data = -np.log(lk)
        
        # Linear regression
        polyfit = np.polyfit(x_data, y_data, 1)
        return polyfit[0]
    
    fd_values = []
    for i in range(len(df)):
        if i < window:
            fd_values.append(np.nan)
        else:
            price_window = df['close'].iloc[i-window:i].values
            if len(price_window) == window:
                fd_values.append(higuchi_fd(price_window))
            else:
                fd_values.append(np.nan)
    
    df['Fractal_Dimension'] = fd_values
    df['Fractal_Dimension'] = df['Fractal_Dimension'].bfill()
    return df

def add_entropy_features(df, window=20):
    """Add Shannon and Sample Entropy features."""
    def shannon_entropy(x):
        bins = np.histogram(x, bins=20)[0]
        probs = bins / len(x)
        probs = probs[probs > 0]  # Remove zeros
        return -np.sum(probs * np.log2(probs))
    
    def sample_entropy(x, m=2, r=0.2):
        """Approximate sample entropy."""
        n = len(x)
        if n < m + 2:
            return np.nan
        
        # Normalize for scale invariance
        x = (x - np.mean(x)) / np.std(x)
        r = r * np.std(x)
        
        # Count template matches for m and m+1
        count_m = 0
        count_m1 = 0
        
        for i in range(n - m):
            template_m = x[i:i+m]
            template_m1 = x[i:i+m+1]
            
            # Count matches of length m
            for j in range(i+1, n - m + 1):
                if np.max(np.abs(template_m - x[j:j+m])) < r:
                    count_m += 1
                    
                    # Count matches of length m+1
                    if j <= n - m - 1:
                        if np.max(np.abs(template_m1 - x[j:j+m+1])) < r:
                            count_m1 += 1
        
        # Calculate sample entropy
        if count_m == 0 or count_m1 == 0:
            return np.nan
        
        return -np.log(count_m1 / count_m)
    
    shannon_values = []
    sample_ent_values = []
    
    for i in range(len(df)):
        if i < window:
            shannon_values.append(np.nan)
            sample_ent_values.append(np.nan)
        else:
            price_window = df['close'].iloc[i-window:i].values
            shannon_values.append(shannon_entropy(price_window))
            sample_ent_values.append(sample_entropy(price_window))
    
    df['Shannon_Entropy'] = shannon_values
    df['Sample_Entropy'] = sample_ent_values
    df['Shannon_Entropy'] = df['Shannon_Entropy'].bfill()
    df['Sample_Entropy'] = df['Sample_Entropy'].bfill()
    return df

def add_hurst_exponent(df, window=100):
    """Calculate Hurst exponent to identify trend persistence or mean reversion."""
    def hurst_exponent(ts, max_lag=20):
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    hurst_values = []
    for i in range(len(df)):
        if i < window:
            hurst_values.append(np.nan)
        else:
            price_window = df['close'].iloc[i-window:i].values
            hurst_values.append(hurst_exponent(price_window))
    
    df['Hurst_Exponent'] = hurst_values
    df['Hurst_Exponent'] = df['Hurst_Exponent'].bfill()
    df['Mean_Reversion_Tendency'] = np.where(df['Hurst_Exponent'] < 0.5, 1, 0)
    df['Trend_Following_Tendency'] = np.where(df['Hurst_Exponent'] > 0.5, 1, 0)
    return df

def add_fourier_transform_features(df, window=100):
    """Extract features using Fast Fourier Transform."""
    fft_magnitudes = []
    dominant_frequencies = []
    spectral_density = []
    
    for i in range(len(df)):
        if i < window:
            fft_magnitudes.append(np.nan)
            dominant_frequencies.append(np.nan)
            spectral_density.append(np.nan)
        else:
            price_window = df['close'].iloc[i-window:i].values
            price_window = price_window - np.mean(price_window)  # Detrend
            
            # Compute FFT
            fft_result = np.fft.fft(price_window)
            fft_freq = np.fft.fftfreq(len(price_window))
            
            # Get magnitudes
            magnitudes = np.abs(fft_result)
            
            # Dominant frequency (excluding DC component)
            idx = np.argmax(magnitudes[1:]) + 1
            
            fft_magnitudes.append(np.mean(magnitudes[1:]))
            dominant_frequencies.append(fft_freq[idx])
            spectral_density.append(np.sum(magnitudes**2) / window)
    
    df['FFT_Magnitude'] = fft_magnitudes
    df['Dominant_Frequency'] = dominant_frequencies
    df['Spectral_Density'] = spectral_density
    
    df[['FFT_Magnitude', 'Dominant_Frequency', 'Spectral_Density']] = \
        df[['FFT_Magnitude', 'Dominant_Frequency', 'Spectral_Density']].bfill()
    return df

def add_price_velocity_acceleration(df):
    """Calculate price velocity and acceleration."""
    df['Price_Velocity'] = df['close'].diff()
    df['Price_Acceleration'] = df['Price_Velocity'].diff()
    
    # Calculate higher moments
    df['Velocity_5d'] = (df['close'] - df['close'].shift(5)) / 5
    df['Acceleration_5d'] = df['Velocity_5d'].diff(5) / 5
    
    # Fill NaNs
    df[['Price_Velocity', 'Price_Acceleration', 'Velocity_5d', 'Acceleration_5d']] = \
        df[['Price_Velocity', 'Price_Acceleration', 'Velocity_5d', 'Acceleration_5d']].bfill()
    return df

def add_statistical_moments(df, window=20):
    """Calculate statistical moments (skewness, kurtosis) over rolling windows."""
    df['Rolling_Skew'] = df['close'].rolling(window=window).skew().bfill()
    df['Rolling_Kurt'] = df['close'].rolling(window=window).kurt().bfill()
    df['Return_Skew'] = df['Price_Change'].rolling(window=window).skew().bfill()
    df['Return_Kurt'] = df['Price_Change'].rolling(window=window).kurt().bfill()
    return df

def add_cross_correlations(df, window=20):
    """Add cross-correlations between price and volume."""
    # Calculate cross-correlation
    price_std = df['close'].rolling(window=window).std()
    volume_std = df['tick_volume'].rolling(window=window).std()
    cov = df['close'].rolling(window=window).cov(df['tick_volume'])
    
    df['Price_Volume_Correlation'] = (cov / (price_std * volume_std)).bfill()
    
    # Add lagged correlations
    price_changes = df['close'].pct_change()
    
    # Correlation with 1-day lagged volume
    df['Price_LaggedVolume_Corr'] = price_changes.rolling(window=window).corr(
        df['tick_volume'].shift(1)).bfill()
    
    return df

def add_peak_trough_features(df, window=50, prominence=0.01):
    """Identify peaks and troughs in the price series."""
    # Find peaks
    prices = df['close'].values
    peak_indices, _ = find_peaks(prices, prominence=prominence*np.mean(prices))
    trough_indices, _ = find_peaks(-prices, prominence=prominence*np.mean(prices))
    
    # Create flags for peaks and troughs
    df['Is_Peak'] = 0
    df['Is_Trough'] = 0
    
    # Use iloc instead of loc for positional indexing
    for idx in peak_indices:
        if idx < len(df):
            df.iloc[idx, df.columns.get_loc('Is_Peak')] = 1
            
    for idx in trough_indices:
        if idx < len(df):
            df.iloc[idx, df.columns.get_loc('Is_Trough')] = 1
    
    # Distance to last peak/trough
    df['Dist_Last_Peak'] = np.nan
    df['Dist_Last_Trough'] = np.nan
    
    last_peak_idx = -np.inf
    last_trough_idx = -np.inf
    
    for i in range(len(df)):
        if df.iloc[i]['Is_Peak'] == 1:
            last_peak_idx = i
        if df.iloc[i]['Is_Trough'] == 1:
            last_trough_idx = i
            
        df.iloc[i, df.columns.get_loc('Dist_Last_Peak')] = i - last_peak_idx if last_peak_idx > -np.inf else np.nan
        df.iloc[i, df.columns.get_loc('Dist_Last_Trough')] = i - last_trough_idx if last_trough_idx > -np.inf else np.nan
    
    # Fill missing values
    df['Dist_Last_Peak'] = df['Dist_Last_Peak'].fillna(window)
    df['Dist_Last_Trough'] = df['Dist_Last_Trough'].fillna(window)
    
    return df

def add_filtered_signals(df):
    """Add low-pass filtered price signals."""
    # Butterworth filter design
    b, a = butter(N=2, Wn=0.1, btype='low')
    
    # Apply filter
    prices = df['close'].values
    filtered_prices = filtfilt(b, a, prices)
    
    df['Filtered_Price'] = filtered_prices
    df['Filter_Ratio'] = df['close'] / df['Filtered_Price']
    
    return df

def add_adaptive_indicators(df):
    """Add indicators that adapt to recent volatility."""
    # Calculate adaptive bands based on recent volatility
    volatility = df['close'].rolling(window=20).std()
    atr = df['ATR']
    
    df['Adaptive_Upper'] = df['close'] + (atr * 2.5 * df['Volatility'] / df['Volatility'].rolling(100).mean())
    df['Adaptive_Lower'] = df['close'] - (atr * 2.5 * df['Volatility'] / df['Volatility'].rolling(100).mean())
    
    # Calculate how far price is within these bands
    df['Adaptive_Position'] = (df['close'] - df['Adaptive_Lower']) / (df['Adaptive_Upper'] - df['Adaptive_Lower'])
    
    # Fill missing values
    df[['Adaptive_Upper', 'Adaptive_Lower', 'Adaptive_Position']] = \
        df[['Adaptive_Upper', 'Adaptive_Lower', 'Adaptive_Position']].bfill()
    
    return df

def add_time_series_features(df):
    """Add time series decomposition features."""
    # Adding hour of day (assuming timestamp index)
    if hasattr(df.index, 'hour'):
        df['Hour_of_Day'] = df.index.hour
        
        # Market session indicators
        df['Is_Asian_Session'] = ((df['Hour_of_Day'] >= 0) & (df['Hour_of_Day'] < 8)).astype(int)
        df['Is_European_Session'] = ((df['Hour_of_Day'] >= 8) & (df['Hour_of_Day'] < 16)).astype(int)
        df['Is_American_Session'] = ((df['Hour_of_Day'] >= 16) & (df['Hour_of_Day'] < 24)).astype(int)
    
    # Add day of month
    if hasattr(df.index, 'day'):
        df['Day_of_Month'] = df.index.day
        
        # Beginning/end of month indicators
        df['Is_Month_Start'] = (df['Day_of_Month'] <= 5).astype(int)
        df['Is_Month_End'] = (df['Day_of_Month'] >= 25).astype(int)
    
    return df

def add_keltner_channels(df, window=20, multiplier=2):
    """Add Keltner Channels."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    ema = typical_price.ewm(span=window, adjust=False).mean()
    atr = df['ATR']
    
    df['Keltner_Middle'] = ema
    df['Keltner_Upper'] = ema + (multiplier * atr)
    df['Keltner_Lower'] = ema - (multiplier * atr)
    
    # Position within channel
    df['Keltner_Position'] = (df['close'] - df['Keltner_Lower']) / (df['Keltner_Upper'] - df['Keltner_Lower'])
    
    # Channel width
    df['Keltner_Width'] = (df['Keltner_Upper'] - df['Keltner_Lower']) / df['Keltner_Middle']
    
    return df

def add_fibonacci_retracements(df, window=100):
    """Add Fibonacci retracement levels."""
    # Find rolling highs and lows
    rolling_high = df['high'].rolling(window=window).max()
    rolling_low = df['low'].rolling(window=window).min()
    
    # Fibonacci levels
    retracement_range = rolling_high - rolling_low
    df['Fib_0'] = rolling_low
    df['Fib_23_6'] = rolling_low + 0.236 * retracement_range
    df['Fib_38_2'] = rolling_low + 0.382 * retracement_range
    df['Fib_50_0'] = rolling_low + 0.5 * retracement_range
    df['Fib_61_8'] = rolling_low + 0.618 * retracement_range
    df['Fib_100'] = rolling_high
    
    # Check if price is near Fibonacci levels
    tolerance = 0.01 * retracement_range
    
    for level in [0, 23.6, 38.2, 50.0, 61.8, 100]:
        col_name = f'Fib_{str(level).replace(".", "_")}'
        indicator_col = f'Near_Fib_{str(level).replace(".", "_")}'
        df[indicator_col] = ((df['close'] >= df[col_name] - tolerance) & 
                             (df['close'] <= df[col_name] + tolerance)).astype(int)
    
    # Fill NaN values
    fib_cols = [col for col in df.columns if col.startswith('Fib_')]
    near_fib_cols = [col for col in df.columns if col.startswith('Near_Fib_')]
    df[fib_cols + near_fib_cols] = df[fib_cols + near_fib_cols].bfill()
    
    return df

def add_ichimoku_cloud(df):
    """Add Ichimoku Cloud indicators."""
    # Conversion Line (Tenkan-sen)
    nine_period_high = df['high'].rolling(window=9).max()
    nine_period_low = df['low'].rolling(window=9).min()
    df['Conversion_Line'] = (nine_period_high + nine_period_low) / 2
    
    # Base Line (Kijun-sen)
    period26_high = df['high'].rolling(window=26).max()
    period26_low = df['low'].rolling(window=26).min()
    df['Base_Line'] = (period26_high + period26_low) / 2
    
    # Leading Span A (Senkou Span A)
    df['Leading_Span_A'] = ((df['Conversion_Line'] + df['Base_Line']) / 2).shift(26)
    
    # Leading Span B (Senkou Span B)
    period52_high = df['high'].rolling(window=52).max()
    period52_low = df['low'].rolling(window=52).min()
    df['Leading_Span_B'] = ((period52_high + period52_low) / 2).shift(26)
    
    # Lagging Span (Chikou Span)
    df['Lagging_Span'] = df['close'].shift(-26)
    
    # Cloud strength
    df['Cloud_Strength'] = df['Leading_Span_A'] - df['Leading_Span_B']
    
    # Price relative to cloud
    df['Above_Cloud'] = ((df['close'] > df['Leading_Span_A']) & 
                          (df['close'] > df['Leading_Span_B'])).astype(int)
    df['Below_Cloud'] = ((df['close'] < df['Leading_Span_A']) & 
                          (df['close'] < df['Leading_Span_B'])).astype(int)
    df['In_Cloud'] = (~df['Above_Cloud'].astype(bool) & ~df['Below_Cloud'].astype(bool)).astype(int)
    
    # Fill NaN values
    ichimoku_cols = ['Conversion_Line', 'Base_Line', 'Leading_Span_A', 'Leading_Span_B', 
                     'Cloud_Strength', 'Above_Cloud', 'Below_Cloud', 'In_Cloud']
    df[ichimoku_cols] = df[ichimoku_cols].bfill()
    
    return df

def add_zscore_features(df, window=20):
    """Add z-score based features."""
    # Price z-score
    df['Price_ZScore'] = (df['close'] - df['close'].rolling(window=window).mean()) / df['close'].rolling(window=window).std()
    
    # Volume z-score
    df['Volume_ZScore'] = (df['tick_volume'] - df['tick_volume'].rolling(window=window).mean()) / df['tick_volume'].rolling(window=window).std()
    
    # RSI z-score
    df['RSI_ZScore'] = (df['RSI'] - df['RSI'].rolling(window=window).mean()) / df['RSI'].rolling(window=window).std()
    
    # Volatility z-score
    df['Volatility_ZScore'] = (df['Volatility'] - df['Volatility'].rolling(window=window).mean()) / df['Volatility'].rolling(window=window).std()
    
    # Extreme z-score flags
    df['Extreme_Price'] = ((df['Price_ZScore'].abs() > 2)).astype(int)
    df['Extreme_Volume'] = ((df['Volume_ZScore'].abs() > 2)).astype(int)
    
    # Fill NaN values
    zscore_cols = ['Price_ZScore', 'Volume_ZScore', 'RSI_ZScore', 'Volatility_ZScore', 'Extreme_Price', 'Extreme_Volume']
    df[zscore_cols] = df[zscore_cols].bfill()
    
    return df

def add_talib_indicators(df):
    """Add various indicators from the TA-Lib library."""
    # Force Index
    df['Force_Index'] = df['close'].diff(1) * df['tick_volume']
    df['Force_Index_EMA13'] = df['Force_Index'].ewm(span=13, adjust=False).mean()
    
    # Money Flow Index
    try:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['tick_volume']
        
        positive_flow = pd.Series(np.where(typical_price > typical_price.shift(1), money_flow, 0), index=df.index)
        negative_flow = pd.Series(np.where(typical_price < typical_price.shift(1), money_flow, 0), index=df.index)
        
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        
        money_ratio = positive_mf / negative_mf
        df['MFI'] = 100 - (100 / (1 + money_ratio))
    except:
        # Fallback to simpler calculation if division issues
        df['MFI'] = 50
    
    # On-Balance Volume (OBV)
    obv = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv[i] = obv[i-1] + df['tick_volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv[i] = obv[i-1] - df['tick_volume'].iloc[i]
        else:
            obv[i] = obv[i-1]
    df['OBV'] = obv
    
    # Chaikin Money Flow
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_volume = money_flow_multiplier * df['tick_volume']
    df['Chaikin_MF'] = money_flow_volume.rolling(window=20).sum() / df['tick_volume'].rolling(window=20).sum()
    
    # Fill NaN values
    talib_cols = ['Force_Index', 'Force_Index_EMA13', 'MFI', 'OBV', 'Chaikin_MF']
    df[talib_cols] = df[talib_cols].bfill()
    
    return df






def engineer_features(df):
    df = df.copy()
    df = add_moving_averages(df)
    df['RSI'] = compute_rsi(df)
    df = categorize_rsi(df)
    df['MACD'], df['Signal'] = compute_macd(df)
    df = generate_macd_signals(df)
    df = add_volatility(df)
    df = add_atr(df)
    df = add_relative_volume(df)
    df = add_volume_change(df)
    df = add_daily_range(df)
    df = add_price_change(df)
    df = add_day_of_week(df)
    df = add_month_of_year(df)
    df = add_wavelet_features(df)
    df = add_lagged_prices(df)
    df = add_bollinger_bands(df)
    df = add_ema(df)
    df = add_roc(df)
    df = add_stochastic_oscillator(df)
    df = add_lagged_returns(df)
    df = add_price_action_patterns(df)
    df = add_cycle_indicator(df)
    df = add_target(df)
    df = add_mean_reversion_indicator(df)
    df = add_momentum_features(df)
    df = add_fractal_dimension(df)
    df = add_entropy_features(df)
    df = add_hurst_exponent(df)
    df = add_fourier_transform_features(df)
    df = add_price_velocity_acceleration(df)
    df = add_statistical_moments(df)
    df = add_cross_correlations(df)
    df = add_peak_trough_features(df)
    df = add_filtered_signals(df)
    df = add_adaptive_indicators(df)
    df = add_time_series_features(df)
    df = add_keltner_channels(df)
    df = add_fibonacci_retracements(df)
    df = add_ichimoku_cloud(df)
    df = add_zscore_features(df)
    df = add_talib_indicators(df)

    df = normalize_features(df)
    df.dropna(inplace=True)
    return df








# --- Command-Line Interface ---
def main():
    parser = argparse.ArgumentParser(description="Feature engineering for XAU/USD forex data.")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV file")
    parser.add_argument('--output', type=str, required=True, help="Path to output CSV file")
    parser.add_argument('--prediction_type', type=str, default='regression',
                        choices=['regression', 'classification'],
                        help="Type of prediction: regression or classification")

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
    else:
        raise ValueError("Input CSV must have a 'timestamp' column for the index.")

    required_cols = ['close', 'high', 'low', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain {required_cols} columns.")

    engineered_df = engineer_features(df, prediction_type=args.prediction_type)
    engineered_df.to_csv(args.output, index=True)
    print(f"Feature-engineered data saved to {args.output}")


if __name__ == "__main__":
    main()