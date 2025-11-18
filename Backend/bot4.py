import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"],allow_credentials=True)

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
        self.timeframe = mt5.TIMEFRAME_H1
        self.num_bars = 3500
        self.lot_size = 0.1
        self.suggested_lot_size = 0.1
        self.error_count = 0
        self.last_error = None
        self.max_loss_percent = 1.2  
        self.position_evaluation_interval = 200  
        self.last_position_refresh = 0
        self.position_cache = []
        self.volatility = 0
        self.max_positions = 5
        self.max_total_risk_percent = 1

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

# --- Train ML Model ---
def train_model(X, y):
    try:
        logger.info("Training Random Forest model...")
        logger.info(f"X shape: {X.shape}, y shape: {len(y)}")
        logger.info(f"X columns: {X.columns.tolist()}")
        
        # Check class distribution
        class_counts = np.bincount(y + 1)  # Shift -1, 0, 1 to 0, 1, 2
        logger.info(f"Class distribution: {class_counts}")
        
        # Calculate class weights inversely proportional to frequencies
        class_weights = {
            -1: len(y) / (class_counts[0] * 3) if class_counts[0] > 0 else 1,
            1: len(y) / (class_counts[2] * 3) if class_counts[2] > 0 else 1
        }
        logger.info(f"Using class weights: {class_weights}")


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(
            n_estimators=500,  # More trees
            max_depth=20,      # Slightly deeper trees
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight=class_weights,  # Custom class weights
            random_state=42,
            n_jobs=-1,  # Use all available cores
            verbose=1
        )

        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        logger.info(f"Cross-validation f1 scores: {scores}, mean: {scores.mean()}")
        model.fit(X_train, y_train)

        # More detailed evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        from sklearn.metrics import classification_report, confusion_matrix
        logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        # Feature importance analysis
        importances = model.feature_importances_
        features = X.columns
        feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        logger.info("Top 10 feature importances:")

        for i, (feature, importance) in enumerate(feature_importance[:10]):
            logger.info(f"{i+1}. {feature}: {importance:.4f}")
        return model
    
    except Exception as e:
        logger.error(f"Exception in train_model: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# --- Check for Open Positions ---
def get_open_positions(symbol: str = None) -> list:
    """Return all open positions with detailed metrics"""
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
            
            # Calculate current profit
            if p.type == mt5.ORDER_TYPE_BUY:
                current_price = tick.ask
            else:
                current_price = tick.bid
                
            profit = (current_price - p.price_open) * p.volume * symbol_info.trade_tick_value
            profit_pct = (profit / p.volume) * 100
            
            # Calculate distance to SL/TP
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
    """Evaluate which positions should be closed"""
    positions = get_open_positions(symbol)
    
    for pos in positions:
        # Check stop loss/take profit
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
        
        # Check maximum loss threshold
        if pos['profit_pct'] <= -state.max_loss_percent:
            pos['should_close'] = True
            pos['close_reason'] = f'Max Loss ({state.max_loss_percent}%) Breached'
        
        # Check time-based exit (close trades after X hours)
        duration = time.time() - pd.to_datetime(pos['open_time']).timestamp()
        hours_open = duration / 3600
        if hours_open > 24:  # Close any position open for more than 24 hours
            pos['should_close'] = True
            pos['close_reason'] = f'Time-based exit ({hours_open:.1f} hours)'
            
    return positions

def close_position(position):
    """Close a single position with improved validation and error handling"""
    try:
        # Validate position data
        required_keys = ['symbol', 'volume', 'type', 'ticket', 'magic', 'close_reason']
        if not all(key in position for key in required_keys):
            logger.error(f"Invalid position data: missing required keys")
            return False

        # Get ticker info
        tick = mt5.symbol_info_tick(position['symbol'])
        if tick is None:
            logger.error(f"Failed to get ticker info for {position['symbol']}")
            return False

        # Check if symbol is available for trading
        symbol_info = mt5.symbol_info(position['symbol'])
        if symbol_info is None or not symbol_info.visible:
            logger.error(f"Symbol {position['symbol']} is not available for trading")
            return False

        # Prepare request
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

        # Send order with retries
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
        else:
            logger.error(f"Failed to close position {position['ticket']} after {max_retries} attempts")
            return False

    except Exception as e:
        logger.error(f"Error closing position: {str(e)}")
        return False
    
# --- for multiple open position management
def calculate_max_position_size(symbol, total_risk_percent):
    """
    Calculate maximum position size based on total account risk
    
    Args:
        symbol (str): Trading symbol
        total_risk_percent (float): Maximum total risk percentage
    
    Returns:
        float: Maximum lot size for a single position
    """
    try:
        account_info = mt5.account_info()
        account_balance = account_info.balance
        
        # Total risk amount
        total_risk_amount = account_balance * (total_risk_percent / 100)
        
        # Risk per trade
        risk_per_trade = total_risk_amount / state.dsmax_positions
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"No symbol info for {symbol}")
            return state.lot_size
        
        # Calculate stop loss in points
        df = fetch_historical_data(symbol, mt5.TIMEFRAME_H1, 300)
        if df is None:
            logger.warning("Failed to calculate volatility for position sizing")
            return state.lot_size
        
        volatility = df['close'].std()
        point_value = symbol_info.point
        stop_loss_points = volatility / point_value
        pip_value = symbol_info.trade_tick_value
        
        # Calculate lot size
        max_lot_size = risk_per_trade / (stop_loss_points * pip_value)
        
        # Apply reasonable constraints
        return max(0.01, min(max_lot_size, 0.5))
    
    except Exception as e:
        logger.error(f"Error in calculate_max_position_size: {str(e)}")
        return state.lot_size



# --- Suggest Lot Size ---
def suggest_lot_size(symbol, df):
    try:
        logger.info("Calculating suggested lot size...")
        
        # Get account information
        account_info = mt5.account_info()
        if account_info is None:
            logger.warning("Failed to get account info, error code: {}".format(mt5.last_error()))
            return state.lot_size
            
        account_balance = account_info.balance
        if account_balance <= 0:
            logger.warning("No account balance available, using default lot size")
            return state.lot_size

        # Get symbol information
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"No symbol info available for {symbol}")
            return state.lot_size

        # Calculate volatility (20-period ATR)
        volatility = df['close'].rolling(window=20).std().iloc[-1]
        
        # Calculate point value and pip size
        point_value = symbol_info.point
        pip_size = 0.0001  # default for most pairs
        if "JPY" in symbol:
            pip_size = 0.01  # adjust for JPY pairs
            
        # Convert volatility to pips
        stop_loss_pips = (volatility / pip_size) * 3  # 3 ATR as stop loss
        
        # Calculate pip value
        pip_value = (pip_size / symbol_info.point) * symbol_info.trade_tick_value
        
        # Risk management parameters
        risk_per_trade = 0.02  # 2% risk per trade
        max_lot_size = 0.5     # absolute maximum lot size
        
        # Calculate position size
        risk_amount = account_balance * risk_per_trade
        lot_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Apply reasonable constraints
        suggested = max(0.01, min(round(lot_size, 2), max_lot_size))
        
        logger.info(f"""
            Lot size calculation:
            - Account balance: {account_balance}
            - Volatility (20-period STD): {volatility:.6f}
            - Stop loss pips: {stop_loss_pips:.1f}
            - Pip value: {pip_value:.2f}
            - Calculated lot size: {lot_size:.2f}
            - Suggested lot size: {suggested}
        """)
        
        return suggested
        
    except Exception as e:
        logger.error(f"Exception in suggest_lot_size: {str(e)}")
        logger.error(traceback.format_exc())
        return state.lot_size

# --- Safe Feature Selection ---
def safe_feature_selection(df):
    try:
        expected_features = [
            # Original features
            'SMA', 'LMA', 'RSI', 'MACD', 'Signal', 'Stoch_K', 'BB_Upper', 'BB_Lower', 'close', 'tick_volume',
            'Volatility', 'ATR', 'Relative_Volume', 'Volume_Change', 'Daily_Range', 'Price_Change',
            'Day_of_Week', 'Month_of_Year', 'Denoised_Price', 'Wavelet_D1', 'Wavelet_D4',
            'Price_t-1', 'Price_t-7', 'Price_t-14', 'BB_Bandwidth', 'EMA_Short', 'EMA_Long', 'ROC',
            'Stoch_D', 'Return_t-1', 'Return_t-7', 'Return_t-14', 'engulfing', 'price_cycle',
            'distance_from_ma', 'momentum','Fractal_Dimension', 'Shannon_Entropy', 'Sample_Entropy', 
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
        
        available_features = [f for f in expected_features if f in df.columns]
        logger.info(f"Available features: {len(available_features)}/{len(expected_features)}: {available_features}")
        
        return available_features
    except Exception as e:
        logger.error(f"Exception in safe_feature_selection: {str(e)}")
        return df.columns.tolist()
    




# --- Smart Trading Strategy ---
def trading_strategy(df, model):
    MIN_CONFIDENCE = 0.65
    RSI_OVERRIDE_CONFIDENCE = 0.70
    try:
        if model is None:
            logger.error("No trained model available")
            return "hold"
        
        current_positions = get_open_positions(state.symbol)
    
        # Check if maximum positions are already open
        if len(current_positions) >= state.max_positions:
            logger.info(f"Maximum positions ({state.max_positions}) already open. Skipping trade.")
            return "hold"
        
        # Safely select features
        available_features = safe_feature_selection(df)
        
        if len(available_features) < 5:
            logger.error(f"Too few features available: {available_features}")
            return "hold"
        
        logger.info(f"Using features: {available_features}")
        
        # Get the latest data point
        X_latest = df[available_features].iloc[-1:].copy()
        
        # Check for NaN values
        if X_latest.isnull().values.any():
            logger.warning("NaN values found in features, filling with 0")
            X_latest = X_latest.fillna(0)
        
        # Debug info
        logger.info(f"X_latest shape: {X_latest.shape}")
        for col in X_latest.columns:
            logger.info(f"Feature {col}: {X_latest[col].values[0]}")
        
        # Make prediction
        prediction = int(model.predict(X_latest)[0])  # Convert numpy.int64 to int
        confidence = float(max(model.predict_proba(X_latest)[0]))  # Convert numpy.float64 to float
        
        # Store results
        state.last_features = dict(zip(X_latest.columns, X_latest.iloc[0]))
        state.last_prediction = prediction
        state.last_confidence = confidence
        state.suggested_lot_size = suggest_lot_size(state.symbol, df)
        
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.2f}, Suggested Lot Size: {state.suggested_lot_size}")
        
        # Check if RSI is available
        if 'RSI' in df.columns:
            rsi = X_latest['RSI'].iloc[0]
            logger.info(f"RSI: {rsi}, Confidence: {confidence}")
            
            if rsi < 30 and confidence > RSI_OVERRIDE_CONFIDENCE:
                logger.info("Overriding to buy due to oversold RSI")
                return "buy"
            elif rsi > 70 and confidence > RSI_OVERRIDE_CONFIDENCE:
                logger.info("Overriding to sell due to overbought RSI")
                return "sell"
        else:
            logger.warning("RSI not available in dataframe")
        
        # Use a lower confidence threshold for now
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
    


#---- Execute Trade V2 -----
def execute_trade(symbol, action, lot_size):
    try:
        logger.info(f"Attempting to execute {action} trade for {symbol} with lot size {lot_size}")
        if has_open_position(symbol):
            logger.info(f"Skipping {action} - position already open")
            return False

        # Initialize critical variables
        sl = None
        tp = None
        price = None
        order_type = None

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return False

        # Get market data
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick data for {symbol}")
            return False

        # Calculate volatility
        df = fetch_historical_data(symbol, mt5.TIMEFRAME_M1, 6000)
        if df is None or df.empty:
            logger.error("Failed to fetch data for volatility calculation")
            return False
        
        volatility = df['close'].std() * 3
        logger.info(f"Calculated volatility: {volatility}")

        # Get minimum stop level in points
        min_stop_points = symbol_info.trade_stops_level
        logger.info(f"Minimum stop level: {min_stop_points} points")
        
        # For forex, point value is typically 0.0001 for most pairs (or 0.01 for JPY pairs)
        point_value = symbol_info.point
        logger.info(f"Point value: {point_value}")
        
        # Minimum stop distance in price terms
        min_stop_distance = min_stop_points * point_value
        logger.info(f"Minimum stop distance: {min_stop_distance}")

        # Set order parameters
        if action == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            
            # Calculate initial SL/TP based on volatility
            raw_sl = price - volatility
            raw_tp = price + 2 * volatility
            
            # Ensure SL respects minimum distance
            required_sl = price - min_stop_distance
            sl = min(raw_sl, required_sl)  # Take the closer one to price (but still valid)
            
            # Ensure TP respects minimum distance
            required_tp = price + min_stop_distance
            tp = max(raw_tp, required_tp)  # Take the farther one from price (but still valid)
            
            logger.info(f"BUY - Price: {price}, Raw SL: {raw_sl}, Required SL: {required_sl}, Final SL: {sl}")
            logger.info(f"BUY - Raw TP: {raw_tp}, Required TP: {required_tp}, Final TP: {tp}")
            
        elif action == "sell":
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            
            # Calculate initial SL/TP based on volatility
            raw_sl = price + volatility
            raw_tp = price - 2 * volatility
            
            # Ensure SL respects minimum distance
            required_sl = price + min_stop_distance
            sl = max(raw_sl, required_sl)  # Take the farther one from price (but still valid)
            
            # Ensure TP respects minimum distance
            required_tp = price - min_stop_distance
            tp = min(raw_tp, required_tp)  # Take the closer one to price (but still valid)
            
            logger.info(f"SELL - Price: {price}, Raw SL: {raw_sl}, Required SL: {required_sl}, Final SL: {sl}")
            logger.info(f"SELL - Raw TP: {raw_tp}, Required TP: {required_tp}, Final TP: {tp}")
        else:
            logger.error("Invalid action, skipping trade")
            return False

        # Validate parameters
        if None in (sl, tp, price, order_type):
            logger.error("Missing critical trade parameters!")
            return False

        # Get symbol info for filling mode
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return False

        # Determine filling mode
        allowed_modes = symbol_info.filling_mode
        filling_mode = mt5.ORDER_FILLING_RETURN
        if allowed_modes & mt5.ORDER_FILLING_IOC:
            filling_mode = mt5.ORDER_FILLING_IOC
        elif allowed_modes & mt5.ORDER_FILLING_FOK:
            filling_mode = mt5.ORDER_FILLING_FOK

        # Convert to native Python types
        try:
            sl = float(sl)
            tp = float(tp)
            price = float(price)
        except TypeError as e:
            logger.error(f"Type conversion failed: {str(e)}")
            return False

        # Build request
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
            "type_filling": filling_mode,  # Use dynamic filling mode
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
    

# --- executing multiple trades ---
def execute_multi_trade(symbol, action, suggested_lot_size):
    """
    Enhanced multi-position management with more flexible position handling
    
    Args:
        symbol (str): Trading symbol
        action (str): Trade action ('buy' or 'sell')
        suggested_lot_size (float): Calculated lot size
    
    Returns:
        bool: Trade execution success
    """
    try:
        # Validate trade action
        if action not in ["buy", "sell"]:
            logger.error(f"Invalid trade action: {action}")
            return False
        
        # Get current open positions for this symbol
        open_positions = get_open_positions(symbol)
        
        # Track total open positions and their direction
        buy_positions = [pos for pos in open_positions if pos.type == mt5.POSITION_TYPE_BUY]
        sell_positions = [pos for pos in open_positions if pos.type == mt5.POSITION_TYPE_SELL]
        
        # Maximum allowed concurrent positions (configurable)
        MAX_CONCURRENT_POSITIONS = 3
        
        # Logic for managing positions based on current market position
        if action == "buy" and len(buy_positions) >= MAX_CONCURRENT_POSITIONS:
            logger.info(f"Max buy positions reached for {symbol}")
            return False
        
        if action == "sell" and len(sell_positions) >= MAX_CONCURRENT_POSITIONS:
            logger.info(f"Max sell positions reached for {symbol}")
            return False
        
        # Calculate maximum position size
        max_lot_size = calculate_max_position_size(
            symbol, 
            state.max_total_risk_percent
        )
        
        # Use the smaller of suggested and max lot size
        lot_size = min(suggested_lot_size, max_lot_size)
        
        # Execute the trade without blocking on existing positions
        success = execute_trade(symbol, action, lot_size)
        
        if success:
            logger.info(f"""
            Multi-Position Trade Details:
            - Symbol: {symbol}
            - Action: {action}
            - Lot Size: {lot_size}
            - Max Lot Size: {max_lot_size}
            - Open Positions: {len(open_positions) + 1}
            - Buy Positions: {len(buy_positions) + (1 if action == 'buy' else 0)}
            - Sell Positions: {len(sell_positions) + (1 if action == 'sell' else 0)}
            """)
        
        return success
    
    except Exception as e:
        logger.error(f"Error in multi-position trade execution: {str(e)}")
        return False



# --- Safe Feature Engineering ---
def safe_engineer_features(df):
    try:
        logger.info("Applying feature engineering")
        df_processed = fe.engineer_features(df)
        logger.info(f"Feature engineering completed, columns: {df_processed.columns.tolist()}")
        return df_processed
    except Exception as e:
        logger.error(f"Exception in feature engineering: {str(e)}")
        logger.error(traceback.format_exc())
        
        # If feature engineering fails, add a simple RSI calculation
        logger.info("Attempting to add basic RSI as fallback")
        try:
            # Calculate basic RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, -1)
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


# --- adaptive trailing for closing early ----
def update_trailing_stop(symbol, trail_amount):
    """Update trailing stop with dynamic trail amount, profit check, and error handling"""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return False

    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Failed to get symbol info for {symbol}")
        return False
    point = symbol_info.point

    for position in positions:
        position_type = position.type
        position_id = position.ticket
        current_sl = position.sl

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get current price for {symbol}")
            continue

        # Calculate profit in points
        price_open = position.price_open
        current_price = tick.bid if position_type == mt5.ORDER_TYPE_BUY else tick.ask
        profit_points = (current_price - price_open) / point if position_type == mt5.ORDER_TYPE_BUY else (price_open - current_price) / point

        # Only trail if in profit by at least 2x the trail amount
        min_profit_to_trail = trail_amount * 2
        if profit_points < min_profit_to_trail:
            continue

        # Calculate new stop loss
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

        # Prepare modification request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": position_id,
            "sl": new_sl,
            "tp": position.tp  # Keep TP the same
        }

        # Send order with retries
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



# --- Trading Loop ---
async def trading_loop():
    last_evaluation = 0
    logger.info("Starting trading loop...")
    try:
        if not initialize_mt5():
            logger.error("Failed to initialize MT5")
            state.running = False
            state.last_error = "MT5 initialization failed"
            return
        
        logger.info("Fetching initial data for model training")
        df = fetch_historical_data(state.symbol, state.timeframe, 1000)
        if df is None:
            logger.error("Failed to fetch initial data")
            state.running = False
            state.last_error = "Failed to fetch initial data"
            return
        
        df = add_volume_column(df)
        df = safe_engineer_features(df)
        
        if 'Target' not in df.columns:
            logger.error("Target column not found in dataframe after feature engineering")
            state.running = False
            state.last_error = "Target column missing"
            return
        
        # Use the same features for training as in trading_strategy
        available_features = safe_feature_selection(df)
        X = df[available_features]
        y = df['Target']
        
        state.model = train_model(X, y)
        if state.model is None:
            logger.error("Failed to train model")
            state.running = False
            state.last_error = "Model training failed"
            return
        
        retrain_interval = 60
        last_retrain = time.time()
        
        while state.running:
            try:
                if state.paused:
                    logger.info("Trading loop paused")
                    await asyncio.sleep(5)
                    continue
                
                logger.info("Entering new trading iteration")
                next_minute = 60 - (time.time() % 60)
                if next_minute < 5:
                    next_minute += 60
                logger.info(f"Waiting {next_minute:.2f} seconds until next trading cycle")
                await asyncio.sleep(next_minute)
                
                df = fetch_historical_data(state.symbol, state.timeframe, state.num_bars)
                if df is None:
                    logger.error("Failed to fetch data in trading loop")
                    await asyncio.sleep(10)
                    continue
                
                df = add_volume_column(df)
                df = safe_engineer_features(df)
                
                action = trading_strategy(df, state.model)
                logger.info(f"Action determined: {action}")
                
                if action in ["buy", "sell"]:
                    success = execute_trade(state.symbol, action, state.suggested_lot_size)
                    logger.info(f"Trade execution {'successful' if success else 'failed'}")
                else:
                    logger.info("No trade executed due to hold signal")
                
                if time.time() - last_retrain > retrain_interval * 60:
                    logger.info("Retraining model...")
                    if 'Target' not in df.columns:
                        logger.error("Target column not found during retraining")
                        continue
                    X = df[available_features]  # Use same features for retraining
                    y = df['Target']
                    new_model = train_model(X, y)
                    if new_model is not None:
                        state.model = new_model
                        last_retrain = time.time()
                        logger.info("Model retraining completed successfully")
                    else:
                        logger.error("Model retraining failed")
                
                if time.time() - last_evaluation > state.position_evaluation_interval:
                    positions = evaluate_positions(state.symbol)
                    for pos in positions:
                        if pos['should_close']:
                            close_position(pos)
                    last_evaluation = time.time()

                if time.time() - state.last_position_refresh > 60:  # Every minute
                    state.position_cache = get_open_positions(state.symbol)
                    state.last_position_refresh = time.time()

                if update_trailing_stop(state.symbol, max(20, state.volatility * 1.5)):
                    logger.info(f"Trailing stops updated successfully")

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
        "last_error": state.last_error
    }

@app.get("/data")
async def get_data():
    logger.info("Data requested")
    df = fetch_historical_data(state.symbol, state.timeframe, state.num_bars)
    if df is None:
        return {"error": "Failed to fetch data"}
    df = add_volume_column(df)
    df = safe_engineer_features(df)
    return df.tail(1).to_dict(orient="records")[0]

@app.get("/symbols", response_model=List[str])
async def get_symbols():
    logger.info("Symbols requested")
    if not mt5.initialize():
        return []
    try:
        symbols = mt5.symbols_get()
        return [s.name for s in symbols if s.visible]
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
        logger.error(f"Symbol {symbol} not available")
        return {"error": f"Symbol {symbol} not available"}
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
        return {"logs": logs[-150:]}  # Return only the last 100 lines
    except Exception as e:
        logger.error(f"Failed to read logs: {str(e)}")
        return {"error": "Failed to fetch logs"}

@app.post("/manual_trade")
async def manual_trade(action: str, lot_size: Optional[float] = None):
    logger.info(f"Manual trade requested: {action}")
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
        # Get all positions if no symbol specified
        positions = evaluate_positions(symbol) if symbol else evaluate_positions()
        
        # Ensure all fields are JSON serializable
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
        positions = evaluate_positions(state.symbol)  # Use evaluate_positions directly
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