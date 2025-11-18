import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import feature_engineering as fe
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import traceback

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    filename="smart_trading_bot.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- FastAPI App ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

# --- Bot State ---
class BotState:
    def __init__(self):
        self.running = False
        self.paused = False
        self.model = None
        self.last_prediction = 0
        self.last_confidence = 0.0
        self.last_features = {}
        self.positions = []
        self.symbol = "EURUSD"
        self.timeframes = [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4]  # Multi-timeframe support
        self.num_bars = 3500
        self.lot_size = 0.1
        self.suggested_lot_size = 0.1
        self.error_count = 0
        self.last_error = None
        self.max_loss_percent = 1.2
        self.position_evaluation_interval = 60  # Reduced for more frequent checks
        self.last_position_refresh = 0
        self.position_cache = []
        self.volatility = 0
        self.max_positions = 5
        self.max_total_risk_percent = 1
        self.last_f1_score = 0.0  # For adaptive retraining
        self.last_volatility = 0.0  # For adaptive retraining

state = BotState()

# --- Initialize MT5 ---
def initialize_mt5(max_retries=3):
    logger.info("Attempting to initialize MT5...")
    for attempt in range(max_retries):
        if mt5.initialize():
            logger.info("MT5 initialized successfully")
            account_info = mt5.account_info()
            logger.info(f"Account Info: {account_info._asdict() if account_info else 'None'}")
            return True
        logger.warning(f"Failed to initialize MT5 (attempt {attempt + 1}/{max_retries})")
        mt5.shutdown()  # Clean up before retry
        time.sleep(5)
    logger.error("Max retries reached. Quitting.")
    return False

# --- Fetch Historical Data ---
def fetch_historical_data(symbol, timeframe, num_bars):
    logger.info(f"Fetching historical data for {symbol}, timeframe: {timeframe}, bars: {num_bars}")
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to fetch data for {symbol}")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        logger.info(f"Data fetched successfully for {symbol}, rows: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Exception in fetch_historical_data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# --- Add Volume Column ---
def add_volume_column(df):
    try:
        logger.info("Adding volume column from tick_volume")
        df['volume'] = df['tick_volume']
        return df
    except Exception as e:
        logger.error(f"Exception in add_volume_column: {str(e)}")
        logger.error(traceback.format_exc())
        return df

# --- Calculate Market Context ---
def calculate_market_context(symbol, df, timeframes):
    """
    Calculate correlations with related pairs and placeholder for economic calendar
    """
    try:
        related_pairs = {
            "EURUSD": ["GBPUSD", "USDJPY", "USDCAD"],
            "GBPUSD": ["EURUSD", "USDJPY", "AUDUSD"],
            # Add more pairs as needed
        }
        context = {}
        if symbol in related_pairs:
            for pair in related_pairs[symbol]:
                for tf in timeframes:
                    pair_df = fetch_historical_data(pair, tf, 100)
                    if pair_df is not None and not pair_df.empty:
                        correlation = df['close'].corr(pair_df['close'])
                        context[f'corr_{pair}_{tf}'] = correlation
                        logger.info(f"Correlation {symbol} vs {pair} (TF {tf}): {correlation:.4f}")
        
        # Placeholder for economic calendar (requires external API)
        # import requests
        # response = requests.get("https://api.example.com/economic_calendar?key=API_KEY")
        # context['economic_event'] = response.json().get('next_event', None)
        
        return context
    except Exception as e:
        logger.error(f"Error in calculate_market_context: {str(e)}")
        return {}

# --- Train ML Model ---
def train_model(X, y):
    try:
        logger.info("Training Random Forest model...")
        X = X.fillna(0)  # Handle NaN values
        y = y.fillna(0)  # Handle NaN in target
        logger.info(f"X shape: {X.shape}, y shape: {len(y)}")
        logger.info(f"X columns: {X.columns.tolist()}")
        
        class_counts = np.bincount(y + 1)
        logger.info(f"Class distribution: {class_counts}")
        class_weights = {
            -1: len(y) / (class_counts[0] * 3) if len(class_counts) > 0 and class_counts[0] > 0 else 1,
            1: len(y) / (class_counts[2] * 3) if len(class_counts) > 2 and class_counts[2] > 0 else 1
        }
        logger.info(f"Using class weights: {class_weights}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        logger.info(f"Cross-validation f1 scores: {scores}, mean: {scores.mean()}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        importances = model.feature_importances_
        features = X.columns
        feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        logger.info("Top 10 feature importances:")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            logger.info(f"{i+1}. {feature}: {importance:.4f}")
        
        return model, f1
    except Exception as e:
        logger.error(f"Exception in train_model: {str(e)}")
        logger.error(traceback.format_exc())
        return None, 0.0

# --- Tune Model Hyperparameters ---
def tune_model(X, y):
    try:
        logger.info("Tuning RandomForest hyperparameters...")
        X = X.fillna(0)  # Handle NaN values
        y = y.fillna(0)  # Handle NaN in target
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 3, 5]
        }
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
        grid_search.fit(X, y)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best F1 score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, grid_search.best_score_
    except Exception as e:
        logger.error(f"Exception in tune_model: {str(e)}")
        return None, 0.0

# --- Check for Open Positions ---
def get_open_positions(symbol: str = None) -> list:
    try:
        if not mt5.initialize():
            logger.error("MT5 not initialized in get_open_positions")
            return []
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        detailed_positions = []
        for p in positions:
            position = p._asdict()
            symbol_info = mt5.symbol_info(p.symbol)
            tick = mt5.symbol_info_tick(p.symbol)
            
            if tick is None or symbol_info is None:
                logger.error(f"Failed to get tick or symbol info for {p.symbol}")
                continue
                
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
        logger.error(f"Error getting positions: {str(e)}")
        return []

def has_open_position(symbol):
    try:
        positions = get_open_positions(symbol)
        return len(positions) > 0
    except Exception as e:
        logger.error(f"Exception in has_open_position: {str(e)}")
        return False

def evaluate_positions(symbol: str = None) -> list:
    positions = get_open_positions(symbol)
    for pos in positions:
        if pos['sl'] is not None:
            if (pos['type'] == 'BUY' and pos['current_price'] <= pos['sl']) or \
               (pos['type'] == 'SELL' and pos['current_price'] >= pos['sl']):
                pos['should_close'] = True
                pos['close_reason'] = 'Stop Loss Hit'
                
        if pos['tp'] is not None:
            if (pos['type'] == 'BUY' and pos['current_price'] >= pos['tp']) or \
               (pos['type'] == 'SELL' and pos['current_price'] <= pos['tp']):
                pos['should_close'] = True
                pos['close_reason'] = 'Take Profit Hit'
        
        if pos['profit_pct'] <= -state.max_loss_percent:
            pos['should_close'] = True
            pos['close_reason'] = f'Max Loss ({state.max_loss_percent}%) Breached'
        
        duration = time.time() - pd.to_datetime(pos['open_time']).timestamp()
        hours_open = duration / 3600
        if hours_open > 24:
            pos['should_close'] = True
            pos['close_reason'] = f'Time-based exit ({hours_open:.1f} hours)'
            
    return positions

def close_position(position):
    try:
        required_keys = ['symbol', 'volume', 'type', 'ticket', 'magic', 'close_reason']
        if not all(key in position for key in required_keys):
            logger.error(f"Invalid position data: missing required keys")
            return False

        tick = mt5.symbol_info_tick(position['symbol'])
        if tick is None:
            logger.error(f"Failed to get ticker info for {position['symbol']}")
            return False

        symbol_info = mt5.symbol_info(position['symbol'])
        if symbol_info is None or not symbol_info.visible:
            logger.error(f"Symbol {position['symbol']} is not available for trading")
            return False

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position['symbol'],
            "volume": position['volume'],
            "type": mt5.ORDER_TYPE_SELL if position['type'] == 'BUY' else mt5.ORDER_TYPE_BUY,
            "position": position['ticket'],
            "price": tick.ask if position['type'] == 'SELL' else tick.bid,
            "deviation": 20,
            "magic": position['magic'],
            "comment": "Auto-close: " + position['close_reason'],
            "type_time": mt5.ORDER_TIME_GTC,
        }

        max_retries = 3
        for attempt in range(max_retries):
            result = mt5.order_send(request)
            if result is None:
                logger.warning(f"Attempt {attempt + 1} failed: No result")
                continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Closed position {position['ticket']} ({position['symbol']}, {position['volume']} lots, price: {request['price']})")
                return True
            logger.warning(f"Attempt {attempt + 1} failed: {result.comment}")
        logger.error(f"Failed to close position {position['ticket']} after {max_retries} attempts")
        return False
    except Exception as e:
        logger.error(f"Error closing position: {str(e)}")
        return False

# --- Calculate Max Position Size ---
def calculate_max_position_size(symbol, total_risk_percent):
    try:
        account_info = mt5.account_info()
        if account_info is None:
            logger.warning("No account info available")
            return state.lot_size
        account_balance = account_info.balance
        
        total_risk_amount = account_balance * (total_risk_percent / 100)
        risk_per_trade = total_risk_amount / state.max_positions
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"No symbol info for {symbol}")
            return state.lot_size
        
        df = fetch_historical_data(symbol, mt5.TIMEFRAME_H1, 300)
        if df is None:
            logger.warning("Failed to calculate volatility for position sizing")
            return state.lot_size
        
        volatility = df['close'].std()
        point_value = symbol_info.point
        stop_loss_points = volatility / point_value
        pip_value = symbol_info.trade_tick_value
        
        max_lot_size = risk_per_trade / (stop_loss_points * pip_value)
        return max(0.01, min(max_lot_size, 0.5))
    except Exception as e:
        logger.error(f"Error in calculate_max_position_size: {str(e)}")
        return state.lot_size

# --- Suggest Lot Size ---
def suggest_lot_size(symbol, df):
    try:
        logger.info("Calculating suggested lot size...")
        account_info = mt5.account_info()
        if account_info is None:
            logger.warning("Failed to get account info")
            return state.lot_size
        account_balance = account_info.balance
        if account_balance <= 0:
            logger.warning("No account balance available")
            return state.lot_size

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"No symbol info available for {symbol}")
            return state.lot_size

        volatility = df['close'].rolling(window=20).std().iloc[-1]
        pip_size = symbol_info.trade_tick_size or symbol_info.point
        stop_loss_pips = (volatility / pip_size) * 3
        pip_value = (pip_size / symbol_info.point) * symbol_info.trade_tick_value
        
        risk_per_trade = 0.02
        max_lot_size = 0.5
        risk_amount = account_balance * risk_per_trade
        lot_size = risk_amount / (stop_loss_pips * pip_value)
        
        suggested = max(0.01, min(round(lot_size, 2), max_lot_size))
        logger.info(f"Lot size calculation: Balance={account_balance}, Volatility={volatility:.6f}, Stop Loss Pips={stop_loss_pips:.1f}, Pip Value={pip_value:.2f}, Suggested={suggested}")
        return suggested
    except Exception as e:
        logger.error(f"Exception in suggest_lot_size: {str(e)}")
        return state.lot_size

# --- Safe Feature Selection ---
def safe_feature_selection(df, timeframe_prefix=""):
    try:
        expected_features = [
            'SMA', 'LMA', 'RSI', 'MACD', 'Signal', 'Stoch_K', 'BB_Upper', 'BB_Lower', 'close', 'tick_volume',
            'Volatility', 'ATR', 'Relative_Volume', 'Volume_Change', 'Daily_Range', 'Price_Change',
            'Day_of_Week', 'Month_of_Year', 'Denoised_Price', 'Wavelet_D1', 'Wavelet_D4',
            'Price_t-1', 'Price_t-7', 'Price_t-14', 'BB_Bandwidth', 'EMA_Short', 'EMA_Long', 'ROC',
            'Stoch_D', 'Return_t-1', 'Return_t-7', 'Return_t-14', 'engulfing', 'price_cycle',
            'distance_from_ma', 'momentum', 'Fractal_Dimension', 'Shannon_Entropy', 'Sample_Entropy',
            'Hurst_Exponent', 'Mean_Reversion_Tendency', 'Trend_Following_Tendency',
            'FFT_Magnitude', 'Dominant_Frequency', 'Spectral_Density',
            'Price_Velocity', 'Price_Acceleration', 'Velocity_5d', 'Acceleration_5d',
            'Rolling_Skew', 'Rolling_Kurt', 'Return_Skew', 'Return_Kurt',
            'Price_Volume_Correlation', 'Price_LaggedVolume_Corr',
            'Is_Peak', 'Is_Trough', 'Dist_Last_Peak', 'Dist_Last_Trough',
            'Filtered_Price', 'Filter_Ratio',
            'Adaptive_Upper', 'Adaptive_Lower', 'Adaptive_Position',
            'Hour_of_Day', 'Is_Asian_Session', 'Is_European_Session', 'Is_American_Session',
            'Day_of_Month', 'Is_Month_Start', 'Is_Month_End',
            'Keltner_Middle', 'Keltner_Upper', 'Keltner_Lower', 'Keltner_Position', 'Keltner_Width',
            'Fib_0', 'Fib_23_6', 'Fib_38_2', 'Fib_50_0', 'Fib_61_8', 'Fib_100',
            'Near_Fib_0', 'Near_Fib_23_6', 'Near_Fib_38_2', 'Near_Fib_50_0', 'Near_Fib_61_8', 'Near_Fib_100',
            'Conversion_Line', 'Base_Line', 'Leading_Span_A', 'Leading_Span_B', 'Cloud_Strength',
            'Above_Cloud', 'Below_Cloud', 'In_Cloud',
            'Price_ZScore', 'Volume_ZScore', 'RSI_ZScore', 'Volatility_ZScore', 'Extreme_Price', 'Extreme_Volume',
            'Force_Index', 'Force_Index_EMA13', 'MFI', 'OBV', 'Chaikin_MF'
        ]
        
        available_features = [f"{timeframe_prefix}{f}" for f in expected_features if f"{timeframe_prefix}{f}" in df.columns]
        logger.info(f"Available features for {timeframe_prefix}: {len(available_features)}/{len(expected_features)}")
        return available_features
    except Exception as e:
        logger.error(f"Exception in safe_feature_selection: {str(e)}")
        return df.columns.tolist()

# --- Smart Trading Strategy ---
def trading_strategy(dfs, model):
    MIN_CONFIDENCE = 0.65
    RSI_OVERRIDE_CONFIDENCE = 0.70
    try:
        if model is None:
            logger.error("No trained model available")
            return "hold"
        
        current_positions = get_open_positions(state.symbol)
        if len(current_positions) >= state.max_positions:
            logger.info(f"Maximum positions ({state.max_positions}) already open. Skipping trade.")
            return "hold"
        
        # Combine features from all timeframes
        X_latest = {}
        for tf, df in dfs.items():
            tf_prefix = f"tf{tf}_"
            available_features = safe_feature_selection(df, tf_prefix)
            if len(available_features) < 5:
                logger.error(f"Too few features available for timeframe {tf}")
                return "hold"
            X_tf = df[available_features].iloc[-1:].copy()
            X_tf = X_tf.fillna(0)
            X_latest.update({col: X_tf[col].values[0] for col in X_tf.columns})
        
        X_latest_df = pd.DataFrame([X_latest])
        logger.info(f"X_latest shape: {X_latest_df.shape}")
        
        # Add market context
        market_context = calculate_market_context(state.symbol, dfs[state.timeframes[1]], state.timeframes)
        X_latest_df.update(market_context)
        
        prediction = int(model.predict(X_latest_df)[0])
        confidence = float(max(model.predict_proba(X_latest_df)[0]))
        
        state.last_features = dict(zip(X_latest_df.columns, X_latest_df.iloc[0]))
        state.last_prediction = prediction
        state.last_confidence = confidence
        state.suggested_lot_size = suggest_lot_size(state.symbol, dfs[state.timeframes[1]])
        
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.2f}, Suggested Lot Size: {state.suggested_lot_size}")
        
        if f"tf{state.timeframes[1]}_RSI" in X_latest_df.columns:
            rsi = X_latest_df[f"tf{state.timeframes[1]}_RSI"].iloc[0]
            logger.info(f"RSI: {rsi}, Confidence: {confidence}")
            if rsi < 30 and confidence > RSI_OVERRIDE_CONFIDENCE:
                logger.info("Overriding to buy due to oversold RSI")
                return "buy"
            elif rsi > 70 and confidence > RSI_OVERRIDE_CONFIDENCE:
                logger.info("Overriding to sell due to overbought RSI")
                return "sell"
        
        if prediction == 1 and confidence > MIN_CONFIDENCE:
            logger.info("Generated buy signal")
            return "buy"
        elif prediction == -1 and confidence > MIN_CONFIDENCE:
            logger.info("Generated sell signal")
            return "sell"
        else:
            logger.info("No trade signal (low confidence or hold)")
            return "hold"
    except Exception as e:
        logger.error(f"Exception in trading_strategy: {str(e)}")
        logger.error(traceback.format_exc())
        return "hold"

# --- Execute Trade ---
def execute_trade(symbol, action, lot_size):
    try:
        logger.info(f"Attempting to execute {action} trade for {symbol} with lot size {lot_size}")
        if has_open_position(symbol):
            logger.info(f"Skipping {action} - position already open")
            return False

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None or not symbol_info.trade_allowed:
            logger.error(f"Symbol {symbol} not tradable")
            return False

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick data for {symbol}")
            return False

        df = fetch_historical_data(symbol, mt5.TIMEFRAME_M1, 6000)
        if df is None or df.empty:
            logger.error("Failed to fetch data for volatility calculation")
            return False
        
        volatility = df['close'].std() * 3
        logger.info(f"Calculated volatility: {volatility}")

        min_stop_points = symbol_info.trade_stops_level
        point_value = symbol_info.point
        min_stop_distance = min_stop_points * point_value

        if action == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            raw_sl = price - volatility
            raw_tp = price + 2 * volatility
            required_sl = price - min_stop_distance
            sl = min(raw_sl, required_sl)
            required_tp = price + min_stop_distance
            tp = max(raw_tp, required_tp)
        elif action == "sell":
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            raw_sl = price + volatility
            raw_tp = price - 2 * volatility
            required_sl = price + min_stop_distance
            sl = max(raw_sl, required_sl)
            required_tp = price - min_stop_distance
            tp = min(raw_tp, required_tp)
        else:
            logger.error("Invalid action, skipping trade")
            return False

        if None in (sl, tp, price, order_type):
            logger.error("Missing critical trade parameters!")
            return False

        allowed_modes = symbol_info.filling_mode
        filling_mode = mt5.ORDER_FILLING_RETURN
        if allowed_modes & mt5.ORDER_FILLING_IOC:
            filling_mode = mt5.ORDER_FILLING_IOC
        elif allowed_modes & mt5.ORDER_FILLING_FOK:
            filling_mode = mt5.ORDER_FILLING_FOK

        sl = float(sl)
        tp = float(tp)
        price = float(price)

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
            "type_filling": filling_mode,
        }

        logger.info(f"Sending trade request: {request}")
        result = mt5.order_send(request)
        
        if result is None:
            logger.error("Order send returned None")
            return False
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to execute {action} order: {result.comment}, Retcode: {result.retcode}")
            return False
            
        logger.info(f"{action.capitalize()} order executed: Price={price}, SL={sl}, TP={tp}, Lot Size={lot_size}")
        return True
    except Exception as e:
        logger.error(f"Exception in execute_trade: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# --- Execute Multi-Trade ---
def execute_multi_trade(symbol, action, suggested_lot_size):
    try:
        if action not in ["buy", "sell"]:
            logger.error(f"Invalid trade action: {action}")
            return False
        
        open_positions = get_open_positions(symbol)
        buy_positions = [pos for pos in open_positions if pos['type'] == 'BUY']
        sell_positions = [pos for pos in open_positions if pos['type'] == 'SELL']
        
        MAX_CONCURRENT_POSITIONS = 3
        if action == "buy" and len(buy_positions) >= MAX_CONCURRENT_POSITIONS:
            logger.info(f"Max buy positions reached for {symbol}")
            return False
        if action == "sell" and len(sell_positions) >= MAX_CONCURRENT_POSITIONS:
            logger.info(f"Max sell positions reached for {symbol}")
            return False
        
        max_lot_size = calculate_max_position_size(symbol, state.max_total_risk_percent)
        lot_size = min(suggested_lot_size, max_lot_size)
        
        success = execute_trade(symbol, action, lot_size)
        
        if success:
            logger.info(f"Multi-Position Trade: Symbol={symbol}, Action={action}, Lot Size={lot_size}, Open Positions={len(open_positions) + 1}")
        return success
    except Exception as e:
        logger.error(f"Error in multi-position trade execution: {str(e)}")
        return False

# --- Safe Feature Engineering ---
def safe_engineer_features(df, timeframe_prefix=""):
    try:
        logger.info(f"Applying feature engineering for {timeframe_prefix}")
        df_processed = fe.engineer_features(df)  # Assumes fe supports timeframe_prefix
        df_processed.columns = [f"{timeframe_prefix}{col}" for col in df_processed.columns]
        logger.info(f"Feature engineering completed, columns: {df_processed.columns.tolist()}")
        return df_processed
    except Exception as e:
        logger.error(f"Exception in feature engineering: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df[f'{timeframe_prefix}RSI'] = 100 - (100 / (1 + rs))
            df[f'{timeframe_prefix}Target'] = np.where(df['close'].shift(-1) > df['close'], 1, -1)
            logger.info("Basic RSI added successfully")
        except Exception as e2:
            logger.error(f"Failed to add basic RSI: {str(e2)}")
        return df

# --- Safe Drop Columns ---
def safe_drop_columns(df, columns_to_drop):
    try:
        logger.info(f"Attempting to drop columns: {columns_to_drop}")
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        logger.info(f"Columns that exist and will be dropped: {existing_columns}")
        if existing_columns:
            return df.drop(columns=existing_columns)
        return df
    except Exception as e:
        logger.error(f"Exception in safe_drop_columns: {str(e)}")
        logger.error(traceback.format_exc())
        return df

# --- Update Trailing Stop ---
def update_trailing_stop(symbol, trail_amount):
    if trail_amount <= 0:
        logger.error(f"Invalid trail_amount: {trail_amount}")
        return False
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return False

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Failed to get symbol info for {symbol}")
        return False
    point = symbol_info.point

    for position in positions:
        position_type = position.type
        position_id = position.ticket
        current_sl = position.sl

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get current price for {symbol}")
            continue

        price_open = position.price_open
        current_price = tick.bid if position_type == mt5.ORDER_TYPE_BUY else tick.ask
        profit_points = (current_price - price_open) / point if position_type == mt5.ORDER_TYPE_BUY else (price_open - current_price) / point

        min_profit_to_trail = trail_amount * 2
        if profit_points < min_profit_to_trail:
            continue

        if position_type == mt5.ORDER_TYPE_BUY:
            new_sl = tick.bid - trail_amount * point
            if current_sl is not None and new_sl <= current_sl:
                continue
        elif position_type == mt5.ORDER_TYPE_SELL:
            new_sl = tick.ask + trail_amount * point
            if current_sl is not None and new_sl >= current_sl:
                continue
        else:
            continue

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": position_id,
            "sl": new_sl,
            "tp": position.tp
        }

        max_retries = 3
        for attempt in range(max_retries):
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Trailing stop for position {position_id} ({symbol}): {current_sl} -> {new_sl}, profit: {profit_points} points")
                break
            logger.warning(f"Attempt {attempt + 1} failed: {result.comment if result else 'No result'}")
        else:
            logger.error(f"Failed to modify SL for position {position_id} after {max_retries} attempts")

    return True

# --- Should Retrain ---
def should_retrain(df, current_f1_score):
    try:
        atr = df['close'].rolling(window=14).std().iloc[-1]
        volatility_change = abs(atr - state.last_volatility) / state.last_volatility if state.last_volatility > 0 else 0
        f1_degradation = state.last_f1_score - current_f1_score if state.last_f1_score > 0 else 0
        
        if volatility_change > 0.3 or f1_degradation > 0.1:
            logger.info(f"Triggering retrain: Volatility change={volatility_change:.2f}, F1 degradation={f1_degradation:.2f}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error in should_retrain: {str(e)}")
        return False

# --- Trading Loop ---
async def trading_loop():
    last_evaluation = 0
    last_tune = 0
    logger.info("Starting trading loop...")
    try:
        if not initialize_mt5():
            logger.error("Failed to initialize MT5")
            state.running = False
            state.last_error = "MT5 initialization failed"
            return
        
        logger.info("Fetching initial data for model training")
        dfs = {}
        for tf in state.timeframes:
            df = fetch_historical_data(state.symbol, tf, 1000)
            if df is None:
                logger.error(f"Failed to fetch initial data for timeframe {tf}")
                state.running = False
                state.last_error = "Failed to fetch initial data"
                return
            df = add_volume_column(df)
            df = safe_engineer_features(df, f"tf{tf}_")
            dfs[tf] = df
        
        available_features = []
        for tf in state.timeframes:
            available_features.extend(safe_feature_selection(dfs[tf], f"tf{tf}_"))
        
        combined_df = pd.concat([dfs[tf] for tf in state.timeframes], axis=1, join='inner')
        X = combined_df[available_features]
        y = dfs[state.timeframes[1]][f"tf{state.timeframes[1]}_Target"]
        
        state.model, state.last_f1_score = tune_model(X, y)
        if state.model is None:
            logger.error("Failed to train initial model")
            state.running = False
            state.last_error = "Model training failed"
            return
        
        state.last_volatility = dfs[state.timeframes[1]]['close'].rolling(window=14).std().iloc[-1]
        
        while state.running:
            try:
                if not mt5.initialize():
                    logger.error("MT5 connection lost, attempting to reinitialize")
                    initialize_mt5()
                    await asyncio.sleep(10)
                    continue
                
                if state.paused:
                    logger.info("Trading loop paused")
                    await asyncio.sleep(5)
                    continue
                
                logger.info("Entering new trading iteration")
                next_minute = 60 - (time.time() % 60)
                if next_minute < 5:
                    next_minute += 60
                await asyncio.sleep(min(next_minute, 300))
                
                dfs = {}
                for tf in state.timeframes:
                    df = fetch_historical_data(state.symbol, tf, state.num_bars)
                    if df is None:
                        logger.error(f"Failed to fetch data for timeframe {tf}")
                        await asyncio.sleep(10)
                        continue
                    df = add_volume_column(df)
                    df = safe_engineer_features(df, f"tf{tf}_")
                    dfs[tf] = df
                
                action = trading_strategy(dfs, state.model)
                logger.info(f"Action determined: {action}")
                
                if action in ["buy", "sell"]:
                    success = execute_multi_trade(state.symbol, action, state.suggested_lot_size)
                    logger.info(f"Trade execution {'successful' if success else 'failed'}")
                
                if time.time() - last_tune > 3600:  # Tune every hour
                    logger.info("Tuning model hyperparameters...")
                    combined_df = pd.concat([dfs[tf] for tf in state.timeframes], axis=1, join='inner')
                    X = combined_df[available_features]
                    y = dfs[state.timeframes[1]][f"tf{state.timeframes[1]}_Target"]
                    new_model, new_f1 = tune_model(X, y)
                    if new_model is not None:
                        state.model = new_model
                        state.last_f1_score = new_f1
                        last_tune = time.time()
                        logger.info("Model tuning completed")
                
                if should_retrain(dfs[state.timeframes[1]], state.last_f1_score):
                    logger.info("Retraining model due to volatility or performance...")
                    combined_df = pd.concat([dfs[tf] for tf in state.timeframes], axis=1, join='inner')
                    X = combined_df[available_features]
                    y = dfs[state.timeframes[1]][f"tf{state.timeframes[1]}_Target"]
                    new_model, new_f1 = train_model(X, y)
                    if new_model is not None:
                        state.model = new_model
                        state.last_f1_score = new_f1
                        state.last_volatility = dfs[state.timeframes[1]]['close'].rolling(window=14).std().iloc[-1]
                        logger.info("Model retraining completed")
                
                if time.time() - last_evaluation > state.position_evaluation_interval:
                    positions = evaluate_positions(state.symbol)
                    for pos in positions:
                        if pos['should_close']:
                            close_position(pos)
                    last_evaluation = time.time()
                    logger.info("Position evaluation completed")
                
                if time.time() - state.last_position_refresh > 60:
                    state.position_cache = get_open_positions(state.symbol)
                    state.last_position_refresh = time.time()
                
                if update_trailing_stop(state.symbol, max(20, state.volatility * 1.5)):
                    logger.info("Trailing stops updated successfully")
            except asyncio.TimeoutError:
                logger.error("Trading loop timeout, continuing")
                continue
            except Exception as e:
                state.error_count += 1
                state.last_error = str(e)
                logger.error(f"Exception in trading loop: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(10)
                if state.error_count % 5 == 0:
                    logger.info("Attempting to reinitialize MT5 due to errors")
                    initialize_mt5()
    
    except Exception as e:
        logger.error(f"Fatal exception in trading loop: {str(e)}")
        logger.error(traceback.format_exc())
        state.running = False
        state.last_error = str(e)

# --- API Endpoints ---
@app.post("/start")
async def start_bot(background_tasks: BackgroundTasks):
    if state.running:
        return {"message": "Bot is already running"}
    logger.info("Bot start requested")
    state.running = True
    state.paused = False
    state.error_count = 0
    state.last_error = None
    background_tasks.add_task(trading_loop)
    return {"message": "Bot started"}

@app.post("/stop")
async def stop_bot():
    if not state.running:
        return {"message": "Bot is already stopped"}
    logger.info("Bot stop requested")
    state.running = False
    return {"message": "Bot stopped"}

@app.post("/pause")
async def pause_bot():
    if not state.running:
        return {"message": "Bot is not running"}
    state.paused = not state.paused
    logger.info(f"Bot {'paused' if state.paused else 'resumed'}")
    return {"message": f"Bot {'paused' if state.paused else 'resumed'}"}

@app.get("/status")
async def get_status():
    logger.info("Status requested")
    return {
        "running": state.running,
        "paused": state.paused,
        "symbol": state.symbol,
        "last_prediction": state.last_prediction,
        "last_confidence": state.last_confidence,
        "suggested_lot_size": state.suggested_lot_size,
        "lot_size": state.lot_size,
        "positions": [p._asdict() for p in state.positions] if state.positions else [],
        "error_count": int(state.error_count),
        "last_error": state.last_error,
        "last_f1_score": state.last_f1_score
    }

@app.get("/data")
async def get_data():
    logger.info("Data requested")
    dfs = {}
    for tf in state.timeframes:
        df = fetch_historical_data(state.symbol, tf, state.num_bars)
        if df is None:
            return {"error": f"Failed to fetch data for timeframe {tf}"}
        df = add_volume_column(df)
        df = safe_engineer_features(df, f"tf{tf}_")
        dfs[tf] = df.tail(1).to_dict(orient="records")[0]
    return dfs

@app.get("/symbols", response_model=List[str])
async def get_symbols():
    logger.info("Symbols requested")
    if not mt5.initialize():
        return []
    try:
        symbols = mt5.symbols_get()
        return [s.name for s in symbols if s.visible and s.trade_allowed]
    except Exception as e:
        logger.error(f"Error getting symbols: {str(e)}")
        return []

class SymbolRequest(BaseModel):
    symbol: str

@app.post("/set_symbol")
async def set_symbol(request: SymbolRequest):
    symbol = request.symbol
    logger.info(f"Setting symbol to {symbol}")
    symbols = await get_symbols()
    if not symbols or symbol not in symbols:
        logger.error(f"Symbol {symbol} not available or not tradable")
        return {"error": f"Symbol {symbol} not available or not tradable"}
    state.symbol = symbol
    return {"message": f"Symbol set to {symbol}"}

@app.post("/set_lot_size")
async def set_lot_size(lot_size: float):
    logger.info(f"Setting lot size to {lot_size}")
    if lot_size <= 0 or lot_size > 10:
        logger.error("Invalid lot size")
        return {"error": "Invalid lot size"}
    state.lot_size = lot_size
    return {"message": f"Lot size set to {lot_size}"}

@app.post("/use_suggested_lot_size")
async def use_suggested_lot_size():
    logger.info(f"Using suggested lot size: {state.suggested_lot_size}")
    state.lot_size = state.suggested_lot_size
    return {"message": f"Lot size set to suggested value: {state.lot_size}"}

@app.get("/logs")
async def get_logs():
    logger.info("Logs requested")
    try:
        with open("smart_trading_bot.log", "r") as log_file:
            logs = log_file.readlines()
        return {"logs": logs[-150:]}
    except Exception as e:
        logger.error(f"Failed to read logs: {str(e)}")
        return {"error": "Failed to fetch logs"}

@app.post("/manual_trade")
async def manual_trade(action: str, lot_size: Optional[float] = None):
    logger.info(f"Manual trade requested: {action}")
    if not mt5.initialize():
        logger.error("MT5 not initialized")
        return {"error": "MT5 not initialized"}
    if action not in ["buy", "sell"]:
        return {"error": "Invalid action. Must be 'buy' or 'sell'"}
    
    if lot_size is None:
        lot_size = state.lot_size
    
    success = execute_trade(state.symbol, action, lot_size)
    if success:
        return {"message": f"Manual {action} trade executed successfully"}
    else:
        return {"error": f"Failed to execute manual {action} trade"}

class PositionResponse(BaseModel):
    positions: List[dict]
    timestamp: float

class ClosePositionResponse(BaseModel):
    success: bool
    message: str

@app.get("/open_positions", response_model=PositionResponse)
async def return_open_positions(symbol: Optional[str] = None):
    try:
        positions = evaluate_positions(symbol) if symbol else evaluate_positions()
        for pos in positions:
            if isinstance(pos.get('open_time'), pd.Timestamp):
                pos['open_time'] = pos['open_time'].isoformat()
            for key, value in list(pos.items()):
                if isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
                    pos[key] = float(value) if isinstance(value, (np.float64, np.float32)) else int(value)
        logger.info(f"Returning {len(positions)} positions")
        return {
            "positions": positions,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error in open_positions: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/close_position/{ticket}", response_model=ClosePositionResponse)
async def api_close_position(ticket: int):
    try:
        positions = evaluate_positions(state.symbol)
        for pos in positions:
            if pos["ticket"] == ticket:
                success = close_position(pos)
                return {"success": success, "message": f"Closed position {ticket}"}
        return {"success": False, "message": "Position not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Run the Bot ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)