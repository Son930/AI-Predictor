import MetaTrader5 as mt5
import pandas as pd
import time
import logging
from bot_state import state, logger

def initialize_mt5(login: int, password: str, server: str, max_retries=3):
    logger.info(f"Attempting to initialize MT5 with login: {login}, server: {server}")
    for attempt in range(max_retries):
        if mt5.initialize(login=login, password=password, server=server):
            logger.info("MT5 initialized successfully")
            account_info = mt5.account_info()
            logger.info(f"Account Info: {account_info._asdict() if account_info else 'None'}")
            return True
        logger.warning(f"Failed to initialize MT5 (attempt {attempt + 1}/{max_retries})")
        time.sleep(5)
    logger.error("Max retries reached. Quitting.")
    return False

def check_mt5_connection():
    if not mt5.terminal_info():
        logger.error("MT5 terminal not running or disconnected")
        return False
    return True

def fetch_historical_data(symbol, timeframe, num_bars):
    CACHE_DURATION = 300  # 5 minutes
    if state.data_cache is not None and time.time() - state.last_data_fetch < CACHE_DURATION:
        logger.info("Using cached data")
        return state.data_cache
    logger.info(f"Fetching historical data for {symbol}, timeframe: {timeframe}, bars: {num_bars}")
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to fetch data for {symbol}")
            # Return empty DataFrame with expected columns
            df = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            return df
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        state.data_cache = df
        state.last_data_fetch = time.time()
        logger.info(f"Data fetched successfully for {symbol}, rows: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Exception in fetch_historical_data: {str(e)}")
        # Return empty DataFrame with expected columns
        df = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

def add_volume_column(df):
    try:
        logger.info("Adding volume column from tick_volume")
        df['volume'] = df['tick_volume']
        return df
    except Exception as e:
        logger.error(f"Exception in add_volume_column: {str(e)}")
        return df

def get_open_positions(symbol: str = None) -> list:
    try:
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        detailed_positions = []
        for p in positions:
            position = p._asdict()
            symbol_info = mt5.symbol_info(p.symbol)
            tick = mt5.symbol_info_tick(p.symbol)
            if p.type == mt5.ORDER_TYPE_BUY:
                current_price = tick.ask
            else:
                current_price = tick.bid
            profit = (current_price - p.price_open) * p.volume * symbol_info.trade_tick_value
            profit_pct = (profit / p.volume) * 100
            sl_distance = abs(p.sl - current_price) if p.sl else None
            tp_distance = abs(p.tp - current_price) if p.tp else None
            detailed_positions.append({
                'ticket': p.ticket,
                'symbol': p.symbol,
                'type': 'BUY' if p.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': p.volume,
                'entry_price': p.price_open,
                'current_price': current_price,
                'profit': profit,
                'profit_pct': profit_pct,
                'sl': p.sl,
                'tp': p.tp,
                'sl_distance': sl_distance,
                'tp_distance': tp_distance,
                'open_time': pd.to_datetime(p.time, unit='s'),
                'magic': p.magic,
                'comment': p.comment,
                'should_close': False,
                'close_reason': None
            })
        return detailed_positions
    except Exception as e:
        logger.error(f"Error in get_open_positions: {str(e)}")
        return []