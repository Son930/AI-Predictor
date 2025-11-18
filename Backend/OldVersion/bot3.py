import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    filename="smart_trading_bot.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Initialize MT5 with Retry Logic ---
def initialize_mt5(max_retries=3):
    for attempt in range(max_retries):
        if mt5.initialize():
            logging.info("MT5 initialized successfully")
            return True
        logging.warning(f"Failed to initialize MT5 (attempt {attempt + 1}/{max_retries})")
        time.sleep(5)
    logging.error("Max retries reached. Quitting.")
    quit()

# --- Fetch Historical Data ---
def fetch_historical_data(symbol, timeframe, num_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        logging.error(f"Failed to fetch data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# --- Add Volume Column ---
def add_volume_column(df):
    df['volume'] = df['tick_volume']
    return df

# --- Basic Feature Engineering ---
def engineer_features(df):
    df['SMA'] = df['close'].rolling(window=20).mean()
    df['LMA'] = df['close'].rolling(window=50).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['Stoch_K'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    df['BB_Std'] = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    
    return df.dropna()

# --- Prepare Training Data ---
def prepare_training_data(df):
    df['Future_Return'] = df['close'].shift(-5) - df['close']
    df['Target'] = np.where(df['Future_Return'] > 0.0005, 1, np.where(df['Future_Return'] < -0.0005, -1, 0))
    features = ['SMA', 'LMA', 'RSI', 'MACD', 'Signal', 'Stoch_K', 'BB_Upper', 'BB_Lower', 'close', 'volume']
    X = df[features]
    y = df['Target']
    return X, y

# --- Train ML Model ---
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy:.2f}")
    
    return model

# --- Check for Open Positions ---
def has_open_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    return len(positions) > 0 if positions else False

# --- Smart Trading Strategy ---
def trading_strategy(df, model):
    features = ['SMA', 'LMA', 'RSI', 'MACD', 'Signal', 'Stoch_K', 'BB_Upper', 'BB_Lower', 'close', 'volume']
    X_latest = df[features].iloc[-1:]  # Keep as DataFrame
    prediction = model.predict(X_latest)[0]
    confidence = max(model.predict_proba(X_latest)[0])
    
    logging.info(f"Features: {dict(zip(features, X_latest.iloc[0]))}")
    logging.info(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
    
    if prediction == 1 and confidence > 0.6:
        return "buy"
    elif prediction == -1 and confidence > 0.6:
        return "sell"
    else:
        logging.info("No trade signal (low confidence or hold)")
        return "hold"

# --- Execute Trade with Dynamic SL/TP ---
def execute_trade(symbol, action, lot_size=0.1):
    if has_open_position(symbol):
        logging.info(f"Skipping {action} - position already open")
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logging.error(f"Failed to get tick data for {symbol}")
        return

    df = fetch_historical_data(symbol, mt5.TIMEFRAME_M1, 20)
    volatility = df['close'].std() * 2
    
    if action == "buy":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = price - volatility
        tp = price + 2 * volatility
    elif action == "sell":
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = price + volatility
        tp = price - 2 * volatility
    else:
        logging.info("No action to execute")
        return

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 234000,
        "comment": "Smart bot trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Failed to execute {action} order: {result.comment}")
    else:
        logging.info(f"{action.capitalize()} order executed: Price={price}, SL={sl}, TP={tp}")

# --- Wait for Next Minute ---
def wait_for_next_minute():
    now = time.time()
    sleep_time = 60 - (now % 60)
    time.sleep(sleep_time)

# --- Main Loop with Retraining ---
def run_bot(symbol, timeframe, num_bars, lot_size=0.1):
    initialize_mt5()
    
    df = fetch_historical_data(symbol, timeframe, 1000)
    if df is None:
        return
    
    df = add_volume_column(df)
    df = engineer_features(df)
    X, y = prepare_training_data(df)
    model = train_model(X, y)
    
    retrain_interval = 60
    last_retrain = time.time()
    
    while True:
        wait_for_next_minute()
        df = fetch_historical_data(symbol, timeframe, num_bars)
        if df is None:
            continue
        
        df = add_volume_column(df)
        df = engineer_features(df)
        
        action = trading_strategy(df, model)
        if action in ["buy", "sell"]:
            execute_trade(symbol, action, lot_size)
        
        if time.time() - last_retrain > retrain_interval * 60:
            logging.info("Retraining model...")
            X, y = prepare_training_data(df)
            model = train_model(X, y)
            last_retrain = time.time()

# --- Run the Bot ---
if __name__ == "__main__":
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M1
    num_bars = 100
    lot_size = 0.1

    run_bot(symbol, timeframe, num_bars, lot_size)