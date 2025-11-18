import pandas as pd
import numpy as np
import time
import asyncio
import logging
import MetaTrader5 as mt5
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from bot_state import state, logger
from mt5_utils import fetch_historical_data, check_mt5_connection, initialize_mt5
import feature_engineering as fe
from mt5_utils import get_open_positions
import traceback
from datetime import datetime, time as dt_time
from collections import deque



# this is the main class for managing positions with advanced logic
# It includes dynamic stop loss and take profit calculations, market momentum analysis,
# and position modification based on performance metrics.
class SmartPositionManager:
    def __init__(self):
        self.position_history = {}  # Track position performance over time
        self.market_momentum = 0.0
        self.volatility_trend = deque(maxlen=10)
        self.profit_zones = {
            'quick_profit': 0.5,    # 0.5% quick profit zone
            'target_profit': 2.0,   # 2% target profit zone
            'super_profit': 5.0     # 5% super profit zone
        }
        
    def analyze_market_momentum(self, df):
        """Analyze current market momentum for position adjustments"""
        if df is None or len(df) < 20:
            return 0.0
            
        try:
            # Calculate momentum indicators
            close_prices = df['close'].tail(20)
            
            # Price momentum (rate of change)
            price_momentum = (close_prices.iloc[-1] - close_prices.iloc[-10]) / close_prices.iloc[-10]
            
            # Volume momentum (if available)
            volume_momentum = 0.0
            if 'tick_volume' in df.columns:
                volumes = df['tick_volume'].tail(20)
                volume_momentum = (volumes.iloc[-5:].mean() - volumes.iloc[-10:-5].mean()) / volumes.iloc[-10:-5].mean()
            
            # RSI-like momentum
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_momentum = (rsi.iloc[-1] - 50) / 50  # Normalize around 50
            
            # Combine momentum indicators
            momentum = (price_momentum * 0.4 + volume_momentum * 0.3 + rsi_momentum * 0.3)
            self.market_momentum = np.clip(momentum, -1.0, 1.0)
            
            logger.debug(f"Market momentum: {self.market_momentum:.3f}")
            return self.market_momentum
            
        except Exception as e:
            logger.error(f"Error calculating market momentum: {str(e)}")
            return 0.0
    
    def calculate_dynamic_sl_tp(self, position, current_price, volatility, momentum):
        """Calculate dynamic SL/TP based on multiple factors"""
        try:
            open_price = position['open_price']
            position_type = position.get('type', position.get('position_type'))
            volume = position['volume']
            ticket = position['ticket']
            
            # Get current profit percentage
            if position_type in ['BUY', 0, mt5.ORDER_TYPE_BUY]:
                profit_pct = ((current_price - open_price) / open_price) * 100
                is_buy = True
            else:
                profit_pct = ((open_price - current_price) / open_price) * 100
                is_buy = False
            
            # Base volatility adjustment
            vol_multiplier = max(1.0, min(3.0, volatility / 0.001))  # Scale volatility
            base_sl_distance = volatility * vol_multiplier * 1.5
            base_tp_distance = volatility * vol_multiplier * 3.0
            
            # Momentum adjustments
            momentum_factor = 1.0 + (momentum * 0.3)  # Max 30% adjustment
            if is_buy:
                momentum_factor = 1.0 + (momentum * 0.3)
            else:
                momentum_factor = 1.0 - (momentum * 0.3)
            
            # Position size adjustment (smaller positions = wider stops)
            size_factor = min(2.0, max(0.5, 0.1 / volume))
            
            # Time-based adjustment
            position_age_hours = self.get_position_age_hours(position)
            time_factor = 1.0 + (position_age_hours / 24) * 0.2  # Gradually widen stops
            
            # Calculate new SL/TP based on profit zones
            new_sl, new_tp = None, None
            
            if profit_pct >= self.profit_zones['super_profit']:
                # In super profit zone - very tight trailing stop
                if is_buy:
                    new_sl = current_price - (base_sl_distance * 0.5 * time_factor)
                    new_tp = current_price + (base_tp_distance * 1.5 * momentum_factor)
                else:
                    new_sl = current_price + (base_sl_distance * 0.5 * time_factor)
                    new_tp = current_price - (base_tp_distance * 1.5 * momentum_factor)
                    
            elif profit_pct >= self.profit_zones['target_profit']:
                # In target profit zone - moderate trailing
                if is_buy:
                    new_sl = current_price - (base_sl_distance * 0.8 * time_factor)
                    new_tp = current_price + (base_tp_distance * momentum_factor)
                else:
                    new_sl = current_price + (base_sl_distance * 0.8 * time_factor)
                    new_tp = current_price - (base_tp_distance * momentum_factor)
                    
            elif profit_pct >= self.profit_zones['quick_profit']:
                # In quick profit zone - move SL to breakeven+
                breakeven_buffer = base_sl_distance * 0.3
                if is_buy:
                    new_sl = max(open_price + breakeven_buffer, 
                               current_price - (base_sl_distance * 1.2 * time_factor))
                    new_tp = current_price + (base_tp_distance * momentum_factor)
                else:
                    new_sl = min(open_price - breakeven_buffer,
                               current_price + (base_sl_distance * 1.2 * time_factor))
                    new_tp = current_price - (base_tp_distance * momentum_factor)
                    
            elif profit_pct <= -2.0:
                # In loss zone - tighten stops to prevent further losses
                emergency_factor = max(0.5, 1.0 + (profit_pct / 10))  # Tighten more with bigger losses
                if is_buy:
                    new_sl = current_price - (base_sl_distance * emergency_factor)
                    new_tp = open_price + (base_tp_distance * 0.5)  # Modest TP
                else:
                    new_sl = current_price + (base_sl_distance * emergency_factor)
                    new_tp = open_price - (base_tp_distance * 0.5)
                    
            else:
                # Normal zone - standard dynamic adjustment
                if is_buy:
                    new_sl = current_price - (base_sl_distance * size_factor * time_factor)
                    new_tp = current_price + (base_tp_distance * momentum_factor)
                else:
                    new_sl = current_price + (base_sl_distance * size_factor * time_factor)
                    new_tp = current_price - (base_tp_distance * momentum_factor)
            
            return new_sl, new_tp, profit_pct
            
        except Exception as e:
            logger.error(f"Error calculating dynamic SL/TP: {str(e)}")
            return None, None, 0.0
    
    def should_modify_position(self, position, new_sl, new_tp, profit_pct):
        """Determine if position should be modified"""
        try:
            current_sl = position.get('sl', position.get('stop_loss'))
            current_tp = position.get('tp', position.get('take_profit'))
            position_type = position.get('type', position.get('position_type'))
            
            # Always modify if no SL/TP set
            if current_sl is None or current_sl == 0.0:
                return True, "No SL set"
            if current_tp is None or current_tp == 0.0:
                return True, "No TP set"
            
            # Check minimum movement threshold to avoid excessive modifications
            min_movement_pips = 5  # Minimum 5 pip movement to modify
            symbol_info = mt5.symbol_info(position['symbol'])
            if symbol_info:
                min_movement = min_movement_pips * symbol_info.point
            else:
                min_movement = 0.0001  # Default for forex
            
            # SL modification logic
            sl_should_modify = False
            sl_reason = ""
            
            if position_type in ['BUY', 0, mt5.ORDER_TYPE_BUY]:
                # For BUY: new SL should be higher (trailing up) or much lower (emergency)
                if new_sl > current_sl + min_movement:
                    sl_should_modify = True
                    sl_reason = "Trailing SL up"
                elif profit_pct <= -2.0 and new_sl < current_sl - min_movement:
                    sl_should_modify = True
                    sl_reason = "Emergency SL tightening"
            else:
                # For SELL: new SL should be lower (trailing down) or much higher (emergency)
                if new_sl < current_sl - min_movement:
                    sl_should_modify = True
                    sl_reason = "Trailing SL down"
                elif profit_pct <= -2.0 and new_sl > current_sl + min_movement:
                    sl_should_modify = True
                    sl_reason = "Emergency SL tightening"
            
            # TP modification logic
            tp_should_modify = False
            tp_reason = ""
            
            if position_type in ['BUY', 0, mt5.ORDER_TYPE_BUY]:
                if abs(new_tp - current_tp) > min_movement:
                    tp_should_modify = True
                    tp_reason = "Dynamic TP adjustment"
            else:
                if abs(new_tp - current_tp) > min_movement:
                    tp_should_modify = True
                    tp_reason = "Dynamic TP adjustment"
            
            # Position age consideration
            position_age = self.get_position_age_hours(position)
            if position_age > 12 and not sl_should_modify:  # Force check after 12 hours
                sl_should_modify = True
                sl_reason = "Time-based adjustment"
            
            should_modify = sl_should_modify or tp_should_modify
            reason = f"{sl_reason}; {tp_reason}".strip("; ")
            
            return should_modify, reason
            
        except Exception as e:
            logger.error(f"Error in should_modify_position: {str(e)}")
            return False, "Error checking modification"
    
    def get_position_age_hours(self, position):
        """Get position age in hours"""
        try:
            open_time = position.get('open_time', position.get('time_open', position.get('time')))
            if open_time is None:
                return 0
                
            if isinstance(open_time, (int, float)):
                open_timestamp = open_time
            else:
                open_timestamp = pd.to_datetime(open_time).timestamp()
            
            return (time.time() - open_timestamp) / 3600
            
        except Exception as e:
            logger.error(f"Error calculating position age: {str(e)}")
            return 0
    
    def modify_position_sltp(self, position, new_sl, new_tp, reason):
        """Modify position SL/TP with enhanced error handling"""
        try:
            ticket = position['ticket']
            symbol = position['symbol']
            
            # Validate SL/TP values
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Cannot get symbol info for {symbol}")
                return False
            
            # Get minimum stop level
            min_stop_level = symbol_info.trade_stops_level * symbol_info.point
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Cannot get tick data for {symbol}")
                return False
            
            # Validate stop levels
            position_type = position.get('type', position.get('position_type'))
            current_price = tick.bid if position_type in ['BUY', 0, mt5.ORDER_TYPE_BUY] else tick.ask
            
            if position_type in ['BUY', 0, mt5.ORDER_TYPE_BUY]:
                if new_sl is not None and current_price - new_sl < min_stop_level:
                    new_sl = current_price - min_stop_level
                if new_tp is not None and new_tp - current_price < min_stop_level:
                    new_tp = current_price + min_stop_level
            else:
                if new_sl is not None and new_sl - current_price < min_stop_level:
                    new_sl = current_price + min_stop_level
                if new_tp is not None and current_price - new_tp < min_stop_level:
                    new_tp = current_price - min_stop_level
            
            # Prepare modification request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": int(ticket),
                "sl": float(new_sl) if new_sl is not None else position.get('sl', 0.0),
                "tp": float(new_tp) if new_tp is not None else position.get('tp', 0.0)
            }
            
            logger.info(f"Modifying position {ticket}: SL={new_sl:.5f}, TP={new_tp:.5f} - {reason}")
            
            # Send modification request
            result = mt5.order_send(request)
            
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Successfully modified position {ticket}")
                
                # Update position history
                if ticket not in self.position_history:
                    self.position_history[ticket] = []
                
                self.position_history[ticket].append({
                    'timestamp': time.time(),
                    'action': 'modify_sltp',
                    'sl': new_sl,
                    'tp': new_tp,
                    'reason': reason,
                    'profit_pct': position.get('profit_pct', 0)
                })
                
                return True
            else:
                error_msg = result.comment if result else "Unknown error"
                logger.warning(f"Failed to modify position {ticket}: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"Error modifying position SL/TP: {str(e)}")
            return False






# Initialize main class for smart trading logic
# This class encapsulates all trading logic, data fetching, model training, and signal generation
# It includes advanced features like dynamic confidence thresholds, market regime detection, and smart position sizing.

class SmartTradingEngine:
    def __init__(self):
        self.last_data_fetch = 0
        self.last_model_update = 0
        self.data_cache = None
        self.feature_cache = None
        self.market_regime = "UNKNOWN"
        self.signal_history = deque(maxlen=10)
        self.trade_cooldown = {}
        self.volatility_window = deque(maxlen=20)
        self.position_manager = SmartPositionManager()
        self.performance_tracker = {
            'trades_today': 0,
            'wins': 0,
            'losses': 0,
            'daily_pnl': 0.0
        }
        
    def is_market_open(self):
        """Check if major forex markets are open"""
        now = datetime.utcnow()
        current_time = now.time()
        weekday = now.weekday()
        
        # Weekend check (Saturday-Sunday)
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        # Major market hours (UTC)
        # London: 08:00-17:00, New York: 13:00-22:00, Tokyo: 00:00-09:00
        market_hours = [
            (dt_time(0, 0), dt_time(9, 0)),    # Tokyo
            (dt_time(8, 0), dt_time(17, 0)),   # London
            (dt_time(13, 0), dt_time(22, 0))   # New York
        ]
        
        return any(start <= current_time <= end for start, end in market_hours)
    
    def detect_market_regime(self, df):
        """Detect current market regime: TRENDING, RANGING, VOLATILE"""
        if df is None or len(df) < 50:
            return "UNKNOWN"
            
        try:
            # Calculate indicators for regime detection
            close_prices = df['close'].tail(50)
            
            # Trend strength (ADX-like calculation)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            
            # Moving averages for trend detection
            ma_20 = close_prices.rolling(20).mean().iloc[-1]
            ma_50 = close_prices.rolling(50).mean().iloc[-1] if len(close_prices) >= 50 else ma_20
            
            # Volatility measure
            volatility = close_prices.pct_change().std() * np.sqrt(1440)  # Annualized
            self.volatility_window.append(volatility)
            avg_volatility = np.mean(self.volatility_window)
            
            # Regime classification
            trend_strength = abs(ma_20 - ma_50) / ma_50 * 100 if ma_50 > 0 else 0
            
            if avg_volatility > 0.02:  # High volatility threshold
                regime = "VOLATILE"
            elif trend_strength > 0.5:  # Strong trend
                regime = "TRENDING"
            else:
                regime = "RANGING"
                
            logger.info(f"Market regime: {regime} (trend_strength: {trend_strength:.3f}, volatility: {avg_volatility:.4f})")
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "UNKNOWN"
    
    def should_fetch_new_data(self):
        """Intelligent data fetching - only when needed"""
        current_time = time.time()
        
        # Fetch data based on timeframe
        if state.timeframe == mt5.TIMEFRAME_M1:
            fetch_interval = 60  # Every minute for M1
        elif state.timeframe == mt5.TIMEFRAME_M5:
            fetch_interval = 300  # Every 5 minutes for M5
        elif state.timeframe == mt5.TIMEFRAME_M15:
            fetch_interval = 900  # Every 15 minutes for M15
        else:
            fetch_interval = 3600  # Default 1 hour
            
        return current_time - self.last_data_fetch > fetch_interval
    
    def calculate_dynamic_confidence_threshold(self):
        """Calculate dynamic confidence threshold based on market conditions and performance"""
        base_threshold = 0.75  # Higher base threshold
        
        # Adjust based on market regime
        regime_adjustments = {
            "TRENDING": -0.05,    # Lower threshold in trending markets
            "RANGING": +0.10,     # Higher threshold in ranging markets
            "VOLATILE": +0.15,    # Much higher threshold in volatile markets
            "UNKNOWN": +0.05
        }
        
        # Adjust based on recent performance
        if self.performance_tracker['trades_today'] > 0:
            win_rate = self.performance_tracker['wins'] / self.performance_tracker['trades_today']
            if win_rate < 0.4:  # Poor performance
                base_threshold += 0.10
            elif win_rate > 0.7:  # Good performance
                base_threshold -= 0.05
        
        # Adjust based on daily PnL
        if self.performance_tracker['daily_pnl'] < -100:  # Down $100
            base_threshold += 0.15  # Be much more selective
        
        threshold = base_threshold + regime_adjustments.get(self.market_regime, 0)
        return max(0.65, min(0.95, threshold))  # Clamp between 65% and 95%
    
    def train_enhanced_model(self, X, y):
        """Train an ensemble model with better parameters"""
        try:
            logger.info("Training enhanced ensemble model...")
            
            if len(np.unique(y)) < 2:
                logger.warning("Insufficient class diversity for training")
                return None
            
            # Class balancing
            class_counts = np.bincount(y + 1)  # Shift to positive indices
            total_samples = len(y)
            
            class_weights = {}
            for i, count in enumerate(class_counts):
                if count > 0:
                    class_weights[i - 1] = total_samples / (len(class_counts) * count)
            
            logger.info(f"Class weights: {class_weights}")
            
            # Create ensemble
            rf_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight=class_weights,
                random_state=42,
                n_jobs=-1
            )
            
            gb_model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
            
            # Train both models
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            rf_model.fit(X_train, y_train)
            gb_model.fit(X_train, y_train)
            
            # Evaluate models
            rf_score = rf_model.score(X_test, y_test)
            gb_score = gb_model.score(X_test, y_test)
            
            logger.info(f"RF accuracy: {rf_score:.3f}, GB accuracy: {gb_score:.3f}")
            
            # Return the better performing model
            if rf_score >= gb_score:
                logger.info("Using Random Forest model")
                return rf_model
            else:
                logger.info("Using Gradient Boosting model")
                return gb_model
                
        except Exception as e:
            logger.error(f"Error training enhanced model: {str(e)}")
            return None
    
    def generate_smart_signal(self, df):
        """Generate trading signals with advanced filtering"""
        try:
            if state.model is None or df is None or len(df) < 50:
                return None, 0.0, "Insufficient data or model"
            
            # Prepare features
            df_features = fe.engineer_features(df)
            if df_features is None or 'Target' not in df_features.columns:
                return None, 0.0, "Feature engineering failed"
            
            X = df_features.drop(columns=['Target', 'Future_Return'], errors='ignore')
            latest_features = X.iloc[-1:].fillna(0)
            
            # Get prediction and confidence
            prediction = state.model.predict(latest_features)[0]
            probabilities = state.model.predict_proba(latest_features)[0]
            confidence = probabilities.max()
            
            # Advanced signal filtering
            dynamic_threshold = self.calculate_dynamic_confidence_threshold()
            
            # Check signal consistency
            self.signal_history.append((prediction, confidence))
            recent_signals = [s[0] for s in list(self.signal_history)[-3:]]
            signal_consistency = len(set(recent_signals)) == 1 if len(recent_signals) >= 3 else False
            
            # Filter conditions
            reasons = []
            
            if confidence < dynamic_threshold:
                reasons.append(f"Low confidence: {confidence:.3f} < {dynamic_threshold:.3f}")
            
            if self.market_regime == "VOLATILE" and confidence < 0.85:
                reasons.append("Volatile market requires higher confidence")
            
            if not signal_consistency and confidence < 0.85:
                reasons.append("Inconsistent recent signals")
            
            # Check trade cooldown
            symbol_cooldown = self.trade_cooldown.get(state.symbol, 0)
            if time.time() < symbol_cooldown:
                reasons.append("Trade cooldown active")
            
            # Daily trade limit
            if self.performance_tracker['trades_today'] >= 10:
                reasons.append("Daily trade limit reached")
            
            if reasons:
                return None, confidence, "; ".join(reasons)
            
            return prediction, confidence, "Signal passed all filters"
            
        except Exception as e:
            logger.error(f"Error generating smart signal: {str(e)}")
            return None, 0.0, f"Signal generation error: {str(e)}"
    
    def calculate_smart_position_size(self, symbol, confidence):
        """Calculate position size based on confidence and market conditions"""
        try:
            # Get symbol info for volume constraints
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Failed to get symbol info for {symbol}")
                return 0.01
            
            # Get volume step and limits
            volume_step = symbol_info.volume_step
            volume_min = symbol_info.volume_min
            volume_max = symbol_info.volume_max
            
            logger.info(f"Volume constraints: min={volume_min}, max={volume_max}, step={volume_step}")
            
            base_size = state.lot_size
            
            # Adjust based on confidence
            confidence_multiplier = min(confidence / 0.75, 1.5)  # Max 1.5x for high confidence
            
            # Adjust based on market regime
            regime_multipliers = {
                "TRENDING": 1.2,
                "RANGING": 0.8,
                "VOLATILE": 0.6,
                "UNKNOWN": 0.7
            }
            
            # Adjust based on recent performance
            performance_multiplier = 1.0
            if self.performance_tracker['trades_today'] > 0:
                win_rate = self.performance_tracker['wins'] / self.performance_tracker['trades_today']
                if win_rate > 0.6:
                    performance_multiplier = 1.1
                elif win_rate < 0.4:
                    performance_multiplier = 0.7
            
            # Account balance check
            account_info = mt5.account_info()
            if account_info:
                risk_per_trade = min(account_info.balance * 0.02, 200)  # Max 2% or $200 per trade
                calculated_size = risk_per_trade / (1000 * base_size)  # More conservative calculation
                base_size = min(base_size, calculated_size)
            
            # Calculate raw size
            raw_size = base_size * confidence_multiplier * regime_multipliers.get(self.market_regime, 1.0) * performance_multiplier
            
            # Round to valid volume step
            if volume_step > 0:
                final_size = round(raw_size / volume_step) * volume_step
            else:
                final_size = round(raw_size, 2)  # Default to 2 decimal places
            
            # Ensure within broker limits
            final_size = max(volume_min, min(final_size, volume_max))
            
            # Safety check - ensure it's at least minimum
            if final_size < volume_min:
                final_size = volume_min
            
            logger.info(f"Position size calculation: base={base_size:.3f}, confidence_mult={confidence_multiplier:.2f}, "
                       f"regime_mult={regime_multipliers.get(self.market_regime, 1.0):.2f}, raw={raw_size:.3f}, final={final_size:.2f}")
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating smart position size: {str(e)}")
            return 0.01  # Safe fallback
    
    def execute_smart_trade(self, signal, confidence, symbol):
        """Execute trade with smart position sizing and risk management"""
        try:
            if signal not in [1, -1]:
                return False
            
            action = "buy" if signal == 1 else "sell"
            smart_lot_size = self.calculate_smart_position_size(symbol, confidence)
            
            # Check existing positions
            open_positions = get_open_positions(symbol)
            if len(open_positions) >= state.max_positions:
                logger.info(f"Max positions ({state.max_positions}) reached for {symbol}")
                return False
            
            # Execute trade
            success = execute_trade(symbol, action, smart_lot_size)
            
            if success:
                # Update performance tracking
                self.performance_tracker['trades_today'] += 1
                
                # Set cooldown (prevents overtrading)
                cooldown_minutes = 15 if self.market_regime == "VOLATILE" else 5
                self.trade_cooldown[symbol] = time.time() + (cooldown_minutes * 60)
                
                logger.info(f"Smart trade executed: {action.upper()} {smart_lot_size} lots of {symbol} "
                           f"(confidence: {confidence:.3f}, regime: {self.market_regime})")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing smart trade: {str(e)}")
            return False





# Create global instance
smart_engine = SmartTradingEngine()

def execute_trade(symbol, action, lot_size):
    """Enhanced trade execution with better error handling and validation"""
    try:
        logger.info(f"Attempting to execute {action} trade for {symbol} with lot size {lot_size}")
        
        if len(get_open_positions(symbol)) >= state.max_positions:
            logger.info(f"Skipping {action} - max positions reached")
            return False

        # Get symbol info and validate
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return False

        # Validate and fix lot size
        volume_step = symbol_info.volume_step
        volume_min = symbol_info.volume_min
        volume_max = symbol_info.volume_max
        
        # Round lot size to valid step
        if volume_step > 0:
            lot_size = round(lot_size / volume_step) * volume_step
        else:
            lot_size = round(lot_size, 2)
        
        # Ensure within limits
        lot_size = max(volume_min, min(lot_size, volume_max))
        
        logger.info(f"Validated lot size: {lot_size} (min: {volume_min}, max: {volume_max}, step: {volume_step})")

        # Get market data
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick data for {symbol}")
            return False

        # Calculate volatility for SL/TP (use smaller dataset for faster calculation)
        df = fetch_historical_data(symbol, mt5.TIMEFRAME_M5, 200)  # Reduced from M1, 1000
        if df is None or df.empty:
            logger.error("Failed to fetch data for volatility calculation")
            return False
        
        # More conservative volatility calculation
        recent_prices = df['close'].tail(50)
        volatility = recent_prices.std() * 2  # Reduced multiplier
        logger.info(f"Calculated volatility: {volatility}")

        # Get minimum stop level
        min_stop_points = symbol_info.trade_stops_level
        point_value = symbol_info.point
        min_stop_distance = min_stop_points * point_value
        
        logger.info(f"Minimum stop level: {min_stop_points} points ({min_stop_distance} price distance)")

        # Set order parameters with more reasonable SL/TP
        if action == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            
            # More conservative SL/TP calculation
            sl_distance = max(volatility, min_stop_distance * 2)  # Ensure reasonable distance
            tp_distance = sl_distance * 2  # 1:2 risk/reward ratio
            
            sl = price - sl_distance
            tp = price + tp_distance
            
        elif action == "sell":
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            
            sl_distance = max(volatility, min_stop_distance * 2)
            tp_distance = sl_distance * 2
            
            sl = price + sl_distance
            tp = price - tp_distance
        else:
            logger.error("Invalid action, skipping trade")
            return False

        # Validate parameters
        if None in (sl, tp, price, order_type):
            logger.error("Missing critical trade parameters!")
            return False

        # Determine filling mode
        allowed_modes = symbol_info.filling_mode
        filling_mode = mt5.ORDER_FILLING_RETURN
        if allowed_modes & mt5.ORDER_FILLING_IOC:
            filling_mode = mt5.ORDER_FILLING_IOC
        elif allowed_modes & mt5.ORDER_FILLING_FOK:
            filling_mode = mt5.ORDER_FILLING_FOK

        # Build and send request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,  # Increased deviation
            "magic": 234000,
            "comment": "Smart bot trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        logger.info(f"Sending trade request: {request}")
        result = mt5.order_send(request)
        
        if result is None:
            logger.error("Order send returned None")
            return False
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to execute {action} order: {result.comment}, Retcode: {result.retcode}")
            
            # Log additional error details
            error_descriptions = {
                10004: "Requote",
                10006: "Request rejected",
                10007: "Request canceled by trader",
                10008: "Order placed",
                10009: "Request completed",
                10010: "Only part of request was completed",
                10011: "Request processing error",
                10012: "Request canceled by timeout",
                10013: "Invalid request",
                10014: "Invalid volume in request",
                10015: "Invalid price in request",
                10016: "Invalid stops in request",
                10017: "Trade is disabled",
                10018: "Market is closed",
                10019: "There is not enough money to complete the request",
                10020: "Prices changed",
                10021: "There are no quotes to process the request",
                10022: "Invalid order expiration date in request",
                10023: "Order state changed",
                10024: "Too frequent requests",
                10025: "No changes in request",
                10026: "Autotrading disabled by server",
                10027: "Autotrading disabled by client terminal",
                10028: "Request locked for processing",
                10029: "Order or position frozen",
                10030: "Invalid order filling type"
            }
            
            error_msg = error_descriptions.get(result.retcode, "Unknown error")
            logger.error(f"MT5 Error {result.retcode}: {error_msg}")
            return False
            
        logger.info(f"{action.upper()} order executed: Price={price:.5f}, SL={sl:.5f}, TP={tp:.5f}, Lot Size={lot_size}")
        return True

    except Exception as e:
        logger.error(f"Exception in execute_trade: {str(e)}")
        logger.error(traceback.format_exc())
        state.error_count += 1
        state.last_error = str(e)
        return False

def evaluate_positions(symbol: str = None):
    """Evaluate which positions should be closed"""
    positions = get_open_positions(symbol)
    
    for pos in positions:
        pos['should_close'] = False
        
        try:
            # Handle different possible field names for opening price
            open_price = None
            if 'price_open' in pos:
                open_price = pos['price_open']
            elif 'open_price' in pos:
                open_price = pos['open_price']
            elif 'price' in pos:
                open_price = pos['price']
            else:
                # Try to get from MT5 position directly if ticket available
                if 'ticket' in pos:
                    mt5_position = mt5.positions_get(ticket=pos['ticket'])
                    if mt5_position and len(mt5_position) > 0:
                        open_price = mt5_position[0].price_open
                
            if open_price is None:
                logger.warning(f"Could not determine open price for position: {pos}")
                continue
            
            # Get current price for profit calculation
            symbol_name = pos.get('symbol', symbol)
            tick = mt5.symbol_info_tick(symbol_name)
            if tick is None:
                logger.warning(f"Could not get current price for {symbol_name}")
                continue
            
            # Determine position type
            position_type = pos.get('type', pos.get('position_type', 'UNKNOWN'))
            
            if position_type in ['BUY', 0, mt5.ORDER_TYPE_BUY]:
                current_price = tick.bid
                profit_pct = ((current_price - open_price) / open_price) * 100
            elif position_type in ['SELL', 1, mt5.ORDER_TYPE_SELL]:
                current_price = tick.ask
                profit_pct = ((open_price - current_price) / open_price) * 100
            else:
                logger.warning(f"Unknown position type: {position_type}")
                continue
            
            pos['current_price'] = current_price
            pos['profit_pct'] = profit_pct
            pos['open_price'] = open_price  # Standardize the field name
            
            # Get SL/TP values with different possible field names
            sl = pos.get('sl', pos.get('stop_loss', pos.get('SL')))
            tp = pos.get('tp', pos.get('take_profit', pos.get('TP')))
            
            # Check stop loss hit
            if sl is not None and sl != 0.0:
                if (position_type in ['BUY', 0, mt5.ORDER_TYPE_BUY] and current_price <= sl) or \
                   (position_type in ['SELL', 1, mt5.ORDER_TYPE_SELL] and current_price >= sl):
                    pos['should_close'] = True
                    pos['close_reason'] = 'Stop Loss Hit'
                    continue
                    
            # Check take profit hit
            if tp is not None and tp != 0.0:
                if (position_type in ['BUY', 0, mt5.ORDER_TYPE_BUY] and current_price >= tp) or \
                   (position_type in ['SELL', 1, mt5.ORDER_TYPE_SELL] and current_price <= tp):
                    pos['should_close'] = True
                    pos['close_reason'] = 'Take Profit Hit'
                    continue
            
            # Check maximum loss threshold
            max_loss_percent = getattr(state, 'max_loss_percent', 5.0)
            if profit_pct <= -max_loss_percent:
                pos['should_close'] = True
                pos['close_reason'] = f'Max Loss ({max_loss_percent}%) Breached'
                continue
            
            # Check time-based exit (close trades after 24 hours)
            open_time = pos.get('open_time', pos.get('time_open', pos.get('time')))
            if open_time is not None:
                try:
                    if isinstance(open_time, (int, float)):
                        # Unix timestamp
                        open_timestamp = open_time
                    else:
                        # String or datetime
                        open_timestamp = pd.to_datetime(open_time).timestamp()
                    
                    duration = time.time() - open_timestamp
                    hours_open = duration / 3600
                    
                    if hours_open > 24:  # Close any position open for more than 24 hours
                        pos['should_close'] = True
                        pos['close_reason'] = f'Time-based exit ({hours_open:.1f} hours)'
                        continue
                        
                except Exception as e:
                    logger.warning(f"Could not parse open time {open_time}: {str(e)}")
                    pass
                    
            logger.debug(f"Position {pos.get('ticket', 'N/A')}: {position_type} {symbol_name}, "
                        f"Profit: {profit_pct:.2f}%, Current: {current_price}, Open: {open_price}")
                        
        except Exception as e:
            logger.error(f"Error evaluating position: {str(e)}")
            logger.error(f"Position data: {pos}")
            continue
            
    return positions

def close_position(position):
    """Close a single position with enhanced error handling"""
    try:
        # Validate essential position data
        required_fields = ['symbol', 'volume', 'ticket']
        missing_fields = [field for field in required_fields if field not in position]
        
        if missing_fields:
            logger.error(f"Invalid position data: missing fields {missing_fields}")
            logger.error(f"Available fields: {list(position.keys())}")
            return False

        symbol = position['symbol']
        volume = position['volume']
        ticket = position['ticket']
        
        # Determine position type with multiple possible field names
        position_type = None
        if 'type' in position:
            position_type = position['type']
        elif 'position_type' in position:
            position_type = position['position_type']
        else:
            # Try to get from MT5 directly
            mt5_position = mt5.positions_get(ticket=ticket)
            if mt5_position and len(mt5_position) > 0:
                position_type = 'BUY' if mt5_position[0].type == mt5.ORDER_TYPE_BUY else 'SELL'
        
        if position_type is None:
            logger.error(f"Could not determine position type for ticket {ticket}")
            return False
        
        # Get current market prices
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get ticker info for {symbol}")
            return False

        # Validate symbol availability
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None or not symbol_info.visible:
            logger.error(f"Symbol {symbol} is not available for trading")
            return False

        # Determine close order type and price
        if position_type in ['BUY', 0, mt5.ORDER_TYPE_BUY]:
            close_order_type = mt5.ORDER_TYPE_SELL
            close_price = tick.bid
        elif position_type in ['SELL', 1, mt5.ORDER_TYPE_SELL]:
            close_order_type = mt5.ORDER_TYPE_BUY
            close_price = tick.ask
        else:
            logger.error(f"Unknown position type: {position_type}")
            return False

        # Get magic number with fallback
        magic = position.get('magic', position.get('magic_number', 234000))

        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": close_order_type,
            "position": int(ticket),
            "price": float(close_price),
            "deviation": 20,
            "magic": int(magic),
            "comment": f"Auto-close: {position.get('close_reason', 'Manual')}",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        logger.info(f"Closing position {ticket}: {position_type} {volume} {symbol} at {close_price}")

        # Send close order with retries
        max_retries = 3
        for attempt in range(max_retries):
            result = mt5.order_send(request)
            
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Successfully closed position {ticket} - {position.get('close_reason', 'Manual')}")
                return True
            
            if result is not None:
                logger.warning(f"Close attempt {attempt + 1} failed: {result.comment} (retcode: {result.retcode})")
            else:
                logger.warning(f"Close attempt {attempt + 1} failed: No result returned")
            
            if attempt < max_retries - 1:  # Don't sleep on last attempt
                time.sleep(1)  # Brief pause between retries
        
        logger.error(f"Failed to close position {ticket} after {max_retries} attempts")
        return False

    except Exception as e:
        logger.error(f"Error closing position: {str(e)}")
        logger.error(f"Position data: {position}")
        logger.error(traceback.format_exc())
        state.error_count += 1
        state.last_error = str(e)
        return False

def update_trailing_stop(symbol, trail_amount):
    """Update trailing stops for profitable positions"""
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            return True

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return False
            
        point = symbol_info.point

        for position in positions:
            try:
                position_type = position.type
                position_id = position.ticket
                current_sl = position.sl

                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
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

                # Send modification request
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": position_id,
                    "sl": new_sl,
                    "tp": position.tp
                }

                result = mt5.order_send(request)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Updated trailing stop for {position_id}: {current_sl} -> {new_sl}")
                else:
                    logger.warning(f"Failed to update trailing stop for {position_id}: {result.comment if result else 'Unknown error'}")
                    
            except Exception as e:
                logger.error(f"Error updating trailing stop for position {position.ticket}: {str(e)}")
                continue

        return True
        
    except Exception as e:
        logger.error(f"Error in update_trailing_stop: {str(e)}")
        return False

async def smart_trading_loop():
    """Intelligent trading loop that adapts to market conditions"""
    logger.info("Starting smart trading loop...")
    
    try:
        # Initialize MT5 connection
        if not check_mt5_connection():
            if hasattr(state, 'mt5_login') and state.mt5_login:
                initialize_mt5(state.mt5_login, state.mt5_password, state.mt5_server)
            else:
                logger.error("MT5 credentials not available")
                return
        
        # Ensure required state variables exist
        if not hasattr(state, 'max_loss_percent'):
            state.max_loss_percent = 5.0
        if not hasattr(state, 'position_evaluation_interval'):
            state.position_evaluation_interval = 30
        if not hasattr(state, 'volatility'):
            state.volatility = 0.01
        
        # Initial model training
        logger.info("Initial model training...")
        df = fetch_historical_data(state.symbol, state.timeframe, state.num_bars)
        if df is not None:
            df_features = fe.engineer_features(df)
            if df_features is not None and 'Target' in df_features.columns:
                X = df_features.drop(columns=['Target', 'Future_Return'], errors='ignore')
                y = df_features['Target']
                state.model = smart_engine.train_enhanced_model(X, y)
                smart_engine.data_cache = df
                smart_engine.last_data_fetch = time.time()
        
        iteration_count = 0
        last_status_log = 0
        
        while state.running:
            try:
                current_time = time.time()
                iteration_count += 1
                
                # Check if paused
                if state.paused:
                    await asyncio.sleep(10)
                    continue
                
                # Check market hours
                if not smart_engine.is_market_open():
                    if current_time - last_status_log > 3600:  # Log once per hour
                        logger.info("Markets closed - waiting...")
                        last_status_log = current_time
                    await asyncio.sleep(300)  # Check every 5 minutes
                    continue
                
                # Smart data fetching
                if smart_engine.should_fetch_new_data():
                    logger.info("Fetching fresh market data...")
                    df = fetch_historical_data(state.symbol, state.timeframe, state.num_bars)
                    
                    if df is not None:
                        smart_engine.data_cache = df
                        smart_engine.last_data_fetch = current_time
                        smart_engine.market_regime = smart_engine.detect_market_regime(df)
                        
                        # Update model less frequently (every 2 hours)
                        if current_time - smart_engine.last_model_update > 7200:
                            logger.info("Updating model with fresh data...")
                            df_features = fe.engineer_features(df)
                            if df_features is not None and 'Target' in df_features.columns:
                                X = df_features.drop(columns=['Target', 'Future_Return'], errors='ignore')
                                y = df_features['Target']
                                new_model = smart_engine.train_enhanced_model(X, y)
                                if new_model is not None:
                                    state.model = new_model
                                    smart_engine.last_model_update = current_time
                
                # Generate and evaluate trading signal
                signal, confidence, reason = smart_engine.generate_smart_signal(smart_engine.data_cache)
                
                state.last_prediction = signal if signal is not None else 0
                state.last_confidence = confidence
                
                if signal is not None:
                    logger.info(f"Trading signal: {signal} (confidence: {confidence:.3f})")
                    success = smart_engine.execute_smart_trade(signal, confidence, state.symbol)
                    if not success:
                        logger.info("Trade execution failed or skipped")
                else:
                    logger.debug(f"No signal: {reason}")
                
                # Position management (every 30 seconds)
                if iteration_count % 6 == 0:  # Every 30 seconds
                    positions = evaluate_positions(state.symbol)
                    df = smart_engine.data_cache
                    momentum = smart_engine.position_manager.analyze_market_momentum(df)
                    volatility = np.mean(smart_engine.volatility_window) if smart_engine.volatility_window else 0.01

                    for pos in positions:
                        # Calculate new SL/TP
                        new_sl, new_tp, profit_pct = smart_engine.position_manager.calculate_dynamic_sl_tp(
                            pos, pos['current_price'], volatility, momentum
                        )
                        should_modify, reason = smart_engine.position_manager.should_modify_position(
                            pos, new_sl, new_tp, profit_pct
                        )
                        if should_modify:
                            smart_engine.position_manager.modify_position_sltp(pos, new_sl, new_tp, reason)

                        if pos.get('should_close', False):
                            if close_position(pos):
                                # Update performance tracking
                                if pos.get('profit_pct', 0) > 0:
                                    smart_engine.performance_tracker['wins'] += 1
                                else:
                                    smart_engine.performance_tracker['losses'] += 1
                                smart_engine.performance_tracker['daily_pnl'] += pos.get('profit', 0)
                    
                    # Update trailing stops
                    trail_amount = max(30, int(np.mean(smart_engine.volatility_window) * 1000) if smart_engine.volatility_window else 30)
                    update_trailing_stop(state.symbol, trail_amount)
                
                # Status logging (every 10 minutes)
                if current_time - last_status_log > 600:
                    trades_today = smart_engine.performance_tracker['trades_today']
                    wins = smart_engine.performance_tracker['wins']
                    losses = smart_engine.performance_tracker['losses']
                    win_rate = wins / trades_today if trades_today > 0 else 0
                    
                    logger.info(f"Status: Regime={smart_engine.market_regime}, Trades={trades_today}, "
                               f"Win Rate={win_rate:.1%}, Positions={len(get_open_positions(state.symbol))}")
                    last_status_log = current_time
                
                # Smart sleep interval based on market activity
                if smart_engine.market_regime == "VOLATILE":
                    sleep_time = 3  # More frequent checks in volatile markets
                elif smart_engine.market_regime == "TRENDING":
                    sleep_time = 5  # Moderate frequency in trending markets
                else:
                    sleep_time = 10  # Less frequent in ranging markets
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                state.error_count += 1
                state.last_error = str(e)
                logger.error(f"Error in smart trading iteration: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(30)  # Longer pause on errors
                
                # Reinitialize on frequent errors
                if state.error_count % 5 == 0:
                    logger.info("Attempting to reinitialize MT5 due to errors")
                    if hasattr(state, 'mt5_login') and state.mt5_login:
                        initialize_mt5(state.mt5_login, state.mt5_password, state.mt5_server)
    
    except Exception as e:
        logger.error(f"Fatal error in smart trading loop: {str(e)}")
        logger.error(traceback.format_exc())
        state.running = False
        state.last_error = str(e)

trading_loop = smart_trading_loop
