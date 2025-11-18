import MetaTrader5 as mt5
import pandas as pd
import time
import logging
import feature_engineering as fe  # Import the feature engineering module

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    filename="trading_bot.log",
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
    df['volume'] = df['tick_volume']  # Use tick_volume as a placeholder
    return df

# --- Check for Open Positions ---
def has_open_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    return len(positions) > 0 if positions else False

# --- Enhanced Trading Strategy ---
def trading_strategy(df):
    # Log key indicator values
    logging.info(f"SMA: {df['SMA'].iloc[-1]}, LMA: {df['LMA'].iloc[-1]}")
    logging.info(f"RSI: {df['RSI'].iloc[-1]}")
    logging.info(f"MACD: {df['MACD'].iloc[-1]}, Signal: {df['Signal'].iloc[-1]}")
    logging.info(f"Stoch_K: {df['Stoch_K'].iloc[-1]}")
    logging.info(f"Close: {df['close'].iloc[-1]}, BB_Lower: {df['BB_Lower'].iloc[-1]}, BB_Upper: {df['BB_Upper'].iloc[-1]}")

    # Buy conditions
    buy_conditions = [
        df['SMA'].iloc[-1] > df['LMA'].iloc[-1],  # SMA > LMA
        df['RSI'].iloc[-1] < 40,  # RSI < 40 (oversold)
        df['MACD'].iloc[-1] > df['Signal'].iloc[-1],  # MACD > Signal
        df['Stoch_K'].iloc[-1] < 20,  # Stochastic %K < 20 (oversold)
        df['close'].iloc[-1] < df['BB_Lower'].iloc[-1],  # Price below Bollinger Lower Band
    ]

    # Sell conditions
    sell_conditions = [
        df['SMA'].iloc[-1] < df['LMA'].iloc[-1],  # SMA < LMA
        df['RSI'].iloc[-1] > 60,  # RSI > 60 (overbought)
        df['MACD'].iloc[-1] < df['Signal'].iloc[-1],  # MACD < Signal
        df['Stoch_K'].iloc[-1] > 80,  # Stochastic %K > 80 (overbought)
        df['close'].iloc[-1] > df['BB_Upper'].iloc[-1],  # Price above Bollinger Upper Band
    ]

    # Log conditions
    logging.info(f"Buy conditions: {buy_conditions}")
    logging.info(f"Sell conditions: {sell_conditions}")

    # Buy if at least 3 buy conditions are met
    if sum(buy_conditions) >= 3:
        return "buy"
    # Sell if at least 3 sell conditions are met
    elif sum(sell_conditions) >= 3:
        return "sell"
    # Hold otherwise
    logging.info("No trade signal generated")
    return "hold"

# --- Execute Trade with SL/TP ---
def execute_trade(symbol, action, lot_size=0.1):
    if has_open_position(symbol):
        logging.info(f"Skipping {action} - position already open")
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logging.error(f"Failed to get tick data for {symbol}")
        return

    if action == "buy":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = price - 50 * mt5.symbol_info(symbol).point  # Stop Loss: 50 pips
        tp = price + 100 * mt5.symbol_info(symbol).point  # Take Profit: 100 pips
    elif action == "sell":
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = price + 50 * mt5.symbol_info(symbol).point  # Stop Loss: 50 pips
        tp = price - 100 * mt5.symbol_info(symbol).point  # Take Profit: 100 pips
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
        "comment": "Python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Failed to execute {action} order: {result.comment}")
    else:
        logging.info(f"{action.capitalize()} order executed successfully: Price={price}, SL={sl}, TP={tp}")

# --- Wait for Next Minute ---
def wait_for_next_minute():
    now = time.time()
    sleep_time = 60 - (now % 60)
    time.sleep(sleep_time)

# --- Main Loop ---
def run_bot(symbol, timeframe, num_bars, lot_size=0.1):
    initialize_mt5()
    while True:
        wait_for_next_minute()  # Sync with minute start
        df = fetch_historical_data(symbol, timeframe, num_bars)
        if df is None:
            continue  # Skip iteration if data fetch fails
        
        df = add_volume_column(df)
        df = fe.engineer_features(df)
        
        action = trading_strategy(df)
        if action in ["buy", "sell"]:
            execute_trade(symbol, action, lot_size)

# --- Run the Bot ---
if __name__ == "__main__":
    symbol = "XAUUSD"  # Replace with your trading symbol
    timeframe = mt5.TIMEFRAME_M1  # 1-minute timeframe
    num_bars = 100  # Number of historical bars to fetch
    lot_size = 0.1  # Lot size for trading

    run_bot(symbol, timeframe, num_bars, lot_size)