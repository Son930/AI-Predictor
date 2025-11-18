import random
import pandas as pd
import numpy as np
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
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
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from symbol_manager import SymbolAnalyzer, StrategyAllocator

class TradingStrategy(Enum):
    """Trading strategy types"""
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    BALANCED = "balanced"
    SCALPING = "scalping"
    SWING = "swing"


class EmotionLevel(Enum):
    """Bot emotion/confidence levels"""
    FEARFUL = "fearful"      # Very conservative, high thresholds
    CAUTIOUS = "cautious"    # Conservative approach
    STANDARD = "standard"    # Normal operation
    CONFIDENT = "confident"  # More risk-taking
    GREEDY = "greedy"       # Highest risk, lowest thresholds


@dataclass
class StrategyConfig:
    """Configuration for each strategy"""
    confidence_threshold: float
    max_positions_per_symbol: int
    position_size_multiplier: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    trade_cooldown_minutes: int
    risk_per_trade_percent: float
    trailing_stop_distance: int
    max_daily_trades: int
    profit_target_adjustment: float


@dataclass
class SymbolConfig:
    """Configuration for each trading symbol"""
    symbol: str
    strategy: TradingStrategy
    emotion: EmotionLevel
    max_positions: int
    enabled: bool = True
    custom_lot_size: Optional[float] = None
    custom_timeframe: Optional[int] = None
    last_trade_time: float = 0.0
    daily_trades: int = 0
    daily_pnl: float = 0.0


class MultiSymbolPositionManager:
    """Advanced position manager for multiple symbols with different strategies"""
    
    def __init__(self):
        self.symbol_positions: Dict[str, List[Dict]] = defaultdict(list)
        self.position_history: Dict[str, List[Dict]] = defaultdict(list)
        self.symbol_momentum: Dict[str, float] = defaultdict(float)
        self.symbol_volatility: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self.profit_zones = {
            'emergency_exit': -3.0,    # Emergency exit at -3%
            'stop_loss': -2.0,         # Standard stop loss
            'breakeven': 0.0,          # Breakeven point
            'quick_profit': 0.8,       # Quick profit zone
            'target_profit': 2.0,      # Target profit zone
            'super_profit': 5.0,       # Super profit zone
            'moon_profit': 10.0        # Exceptional profit zone
        }
        
    def analyze_symbol_momentum(self, symbol: str, df: pd.DataFrame) -> float:
        """Analyze momentum for specific symbol"""
        if df is None or len(df) < 20:
            return 0.0
            
        try:
            close_prices = df['close'].tail(20)
            
            # Multi-timeframe momentum analysis
            short_momentum = (close_prices.iloc[-1] - close_prices.iloc[-5]) / close_prices.iloc[-5]
            medium_momentum = (close_prices.iloc[-1] - close_prices.iloc[-10]) / close_prices.iloc[-10]
            long_momentum = (close_prices.iloc[-1] - close_prices.iloc[-20]) / close_prices.iloc[-20]
            
            # Volume analysis if available
            volume_momentum = 0.0
            if 'tick_volume' in df.columns:
                volumes = df['tick_volume'].tail(20)
                recent_vol = volumes.iloc[-5:].mean()
                historical_vol = volumes.iloc[-20:-5].mean()
                volume_momentum = (recent_vol - historical_vol) / historical_vol if historical_vol > 0 else 0
            
            # Volatility momentum
            volatility = close_prices.pct_change().std()
            self.symbol_volatility[symbol].append(volatility)
            
            # Combine momentum indicators with weights
            combined_momentum = (
                short_momentum * 0.5 +
                medium_momentum * 0.3 +
                long_momentum * 0.2 +
                volume_momentum * 0.1
            )
            
            self.symbol_momentum[symbol] = np.clip(combined_momentum, -1.0, 1.0)
            
            logger.debug(f"{symbol} momentum: {self.symbol_momentum[symbol]:.3f}")
            return self.symbol_momentum[symbol]
            
        except Exception as e:
            logger.error(f"Error calculating momentum for {symbol}: {str(e)}")
            return 0.0
    
    def calculate_strategy_based_sltp(self, position: Dict, symbol_config: SymbolConfig, 
                                    current_price: float, volatility: float, momentum: float) -> Tuple[Optional[float], Optional[float], float]:
        """Calculate SL/TP based on strategy and emotion"""
        try:
            strategy_configs = self._get_strategy_configs()
            config = strategy_configs[symbol_config.strategy][symbol_config.emotion]
            
            open_price = position['open_price']
            position_type = position.get('type', position.get('position_type'))
            is_buy = position_type in ['BUY', 0, mt5.ORDER_TYPE_BUY]
            
            # Calculate current profit
            if is_buy:
                profit_pct = ((current_price - open_price) / open_price) * 100
            else:
                profit_pct = ((open_price - current_price) / open_price) * 100
            
            # Base distances adjusted by strategy and emotion
            base_sl_distance = volatility * config.stop_loss_multiplier
            base_tp_distance = volatility * config.take_profit_multiplier * config.profit_target_adjustment
            
            # Momentum adjustments
            momentum_factor = 1.0 + (momentum * 0.4 if is_buy else momentum * -0.4)
            
            # Position age factor
            position_age_hours = self._get_position_age_hours(position)
            time_decay = 1.0 + (position_age_hours / 48) * 0.1  # Gradual widening over 48 hours
            
            # Strategy-specific logic
            new_sl, new_tp = self._apply_strategy_logic(
                symbol_config.strategy, symbol_config.emotion, position, 
                current_price, profit_pct, base_sl_distance, base_tp_distance,
                momentum_factor, time_decay, is_buy
            )
            
            return new_sl, new_tp, profit_pct
            
        except Exception as e:
            logger.error(f"Error calculating strategy-based SL/TP: {str(e)}")
            return None, None, 0.0
    
    def _apply_strategy_logic(self, strategy: TradingStrategy, emotion: EmotionLevel,
                            position: Dict, current_price: float, profit_pct: float,
                            base_sl_distance: float, base_tp_distance: float,
                            momentum_factor: float, time_decay: float, is_buy: bool) -> Tuple[Optional[float], Optional[float]]:
        """Apply specific strategy and emotion logic"""
        
        open_price = position['open_price']
        
        # Emotion-based adjustments
        emotion_multipliers = {
            EmotionLevel.FEARFUL: {'sl': 0.5, 'tp': 0.8},      # Tighter stops, conservative targets
            EmotionLevel.CAUTIOUS: {'sl': 0.7, 'tp': 1.0},     # Slightly tighter stops
            EmotionLevel.STANDARD: {'sl': 1.0, 'tp': 1.2},     # Standard approach
            EmotionLevel.CONFIDENT: {'sl': 1.3, 'tp': 1.5},    # Wider stops, bigger targets
            EmotionLevel.GREEDY: {'sl': 1.8, 'tp': 2.0}        # Widest stops, biggest targets
        }
        
        emotion_mult = emotion_multipliers[emotion]
        
        # Strategy-specific implementations
        if strategy == TradingStrategy.AGGRESSIVE:
            return self._aggressive_strategy(
                current_price, open_price, profit_pct, base_sl_distance, base_tp_distance,
                emotion_mult, momentum_factor, time_decay, is_buy
            )
            
        elif strategy == TradingStrategy.PASSIVE:
            return self._passive_strategy(
                current_price, open_price, profit_pct, base_sl_distance, base_tp_distance,
                emotion_mult, momentum_factor, time_decay, is_buy
            )
            
        elif strategy == TradingStrategy.SCALPING:
            return self._scalping_strategy(
                current_price, open_price, profit_pct, base_sl_distance, base_tp_distance,
                emotion_mult, momentum_factor, time_decay, is_buy
            )
            
        elif strategy == TradingStrategy.SWING:
            return self._swing_strategy(
                current_price, open_price, profit_pct, base_sl_distance, base_tp_distance,
                emotion_mult, momentum_factor, time_decay, is_buy
            )
            
        else:  # BALANCED
            return self._balanced_strategy(
                current_price, open_price, profit_pct, base_sl_distance, base_tp_distance,
                emotion_mult, momentum_factor, time_decay, is_buy
            )
    
    def _aggressive_strategy(self, current_price, open_price, profit_pct, base_sl, base_tp,
                           emotion_mult, momentum_factor, time_decay, is_buy):
        """Aggressive strategy: Quick profits, tight trailing, high frequency"""
        
        # Aggressive profit taking
        if profit_pct >= self.profit_zones['quick_profit']:
            # Very tight trailing stop in profit
            trail_factor = 0.3 * emotion_mult['sl']
            if is_buy:
                new_sl = current_price - (base_sl * trail_factor)
                new_tp = current_price + (base_tp * momentum_factor * 0.8)  # Closer TP
            else:
                new_sl = current_price + (base_sl * trail_factor)
                new_tp = current_price - (base_tp * momentum_factor * 0.8)
        
        elif profit_pct <= self.profit_zones['stop_loss']:
            # Quick exit on losses
            emergency_factor = 0.6 * emotion_mult['sl']
            if is_buy:
                new_sl = current_price - (base_sl * emergency_factor)
                new_tp = open_price + (base_tp * 0.5)
            else:
                new_sl = current_price + (base_sl * emergency_factor)
                new_tp = open_price - (base_tp * 0.5)
        
        else:
            # Normal aggressive positioning
            if is_buy:
                new_sl = current_price - (base_sl * emotion_mult['sl'] * time_decay)
                new_tp = current_price + (base_tp * emotion_mult['tp'] * momentum_factor)
            else:
                new_sl = current_price + (base_sl * emotion_mult['sl'] * time_decay)
                new_tp = current_price - (base_tp * emotion_mult['tp'] * momentum_factor)
        
        return new_sl, new_tp
    
    def _passive_strategy(self, current_price, open_price, profit_pct, base_sl, base_tp,
                         emotion_mult, momentum_factor, time_decay, is_buy):
        """Passive strategy: Let profits run, wider stops, patient approach"""
        
        if profit_pct >= self.profit_zones['target_profit']:
            # Let profits run with loose trailing
            trail_factor = 1.2 * emotion_mult['sl']
            if is_buy:
                new_sl = current_price - (base_sl * trail_factor)
                new_tp = current_price + (base_tp * emotion_mult['tp'] * 1.8)  # Distant TP
            else:
                new_sl = current_price + (base_sl * trail_factor)
                new_tp = current_price - (base_tp * emotion_mult['tp'] * 1.8)
        
        elif profit_pct >= self.profit_zones['breakeven']:
            # Move to breakeven slowly
            if is_buy:
                new_sl = max(open_price, current_price - (base_sl * emotion_mult['sl'] * 1.5))
                new_tp = current_price + (base_tp * emotion_mult['tp'] * momentum_factor)
            else:
                new_sl = min(open_price, current_price + (base_sl * emotion_mult['sl'] * 1.5))
                new_tp = current_price - (base_tp * emotion_mult['tp'] * momentum_factor)
        
        else:
            # Wide stops, patient approach
            wide_factor = 1.5 * emotion_mult['sl'] * time_decay
            if is_buy:
                new_sl = current_price - (base_sl * wide_factor)
                new_tp = current_price + (base_tp * emotion_mult['tp'] * 1.5)
            else:
                new_sl = current_price + (base_sl * wide_factor)
                new_tp = current_price - (base_tp * emotion_mult['tp'] * 1.5)
        
        return new_sl, new_tp
    
    def _scalping_strategy(self, current_price, open_price, profit_pct, base_sl, base_tp,
                          emotion_mult, momentum_factor, time_decay, is_buy):
        """Scalping strategy: Very tight stops, quick small profits"""
        
        # Scalping uses much tighter ranges
        tight_sl = base_sl * 0.4 * emotion_mult['sl']
        tight_tp = base_tp * 0.6 * emotion_mult['tp']
        
        if profit_pct >= 0.3:  # Take profit at 0.3%
            # Immediate profit taking
            if is_buy:
                new_sl = current_price - (tight_sl * 0.5)  # Very tight trail
                new_tp = current_price + (tight_tp * 0.3)  # Quick exit
            else:
                new_sl = current_price + (tight_sl * 0.5)
                new_tp = current_price - (tight_tp * 0.3)
        
        elif profit_pct <= -0.5:  # Quick stop at -0.5%
            # Immediate exit
            if is_buy:
                new_sl = current_price - (tight_sl * 0.2)
                new_tp = open_price + (tight_tp * 0.2)
            else:
                new_sl = current_price + (tight_sl * 0.2)
                new_tp = open_price - (tight_tp * 0.2)
        
        else:
            # Standard tight scalping levels
            if is_buy:
                new_sl = current_price - tight_sl
                new_tp = current_price + tight_tp
            else:
                new_sl = current_price + tight_sl
                new_tp = current_price - tight_tp
        
        return new_sl, new_tp
    
    def _swing_strategy(self, current_price, open_price, profit_pct, base_sl, base_tp,
                       emotion_mult, momentum_factor, time_decay, is_buy):
        """Swing strategy: Wide stops, big targets, hold for trends"""
        
        # Swing trading uses wider ranges
        wide_sl = base_sl * 2.0 * emotion_mult['sl'] * time_decay
        wide_tp = base_tp * 2.5 * emotion_mult['tp']
        
        if profit_pct >= self.profit_zones['super_profit']:
            # Trail with wide stops
            trail_factor = 1.5 * emotion_mult['sl']
            if is_buy:
                new_sl = current_price - (base_sl * trail_factor)
                new_tp = current_price + (wide_tp * momentum_factor)
            else:
                new_sl = current_price + (base_sl * trail_factor)
                new_tp = current_price - (wide_tp * momentum_factor)
        
        elif profit_pct >= self.profit_zones['target_profit']:
            # Partial trailing
            if is_buy:
                new_sl = max(open_price + (base_sl * 0.5), current_price - wide_sl)
                new_tp = current_price + (wide_tp * momentum_factor)
            else:
                new_sl = min(open_price - (base_sl * 0.5), current_price + wide_sl)
                new_tp = current_price - (wide_tp * momentum_factor)
        
        else:
            # Wide stops for trend following
            if is_buy:
                new_sl = current_price - wide_sl
                new_tp = current_price + (wide_tp * momentum_factor)
            else:
                new_sl = current_price + wide_sl
                new_tp = current_price - (wide_tp * momentum_factor)
        
        return new_sl, new_tp
    
    def _balanced_strategy(self, current_price, open_price, profit_pct, base_sl, base_tp,
                          emotion_mult, momentum_factor, time_decay, is_buy):
        """Balanced strategy: Moderate approach, adaptable"""
        
        if profit_pct >= self.profit_zones['target_profit']:
            # Moderate trailing
            trail_factor = 0.8 * emotion_mult['sl']
            if is_buy:
                new_sl = current_price - (base_sl * trail_factor)
                new_tp = current_price + (base_tp * emotion_mult['tp'] * momentum_factor)
            else:
                new_sl = current_price + (base_sl * trail_factor)
                new_tp = current_price - (base_tp * emotion_mult['tp'] * momentum_factor)
        
        elif profit_pct >= self.profit_zones['quick_profit']:
            # Move to breakeven
            buffer = base_sl * 0.3
            if is_buy:
                new_sl = max(open_price + buffer, current_price - (base_sl * emotion_mult['sl']))
                new_tp = current_price + (base_tp * emotion_mult['tp'] * momentum_factor)
            else:
                new_sl = min(open_price - buffer, current_price + (base_sl * emotion_mult['sl']))
                new_tp = current_price - (base_tp * emotion_mult['tp'] * momentum_factor)
        
        else:
            # Standard approach
            if is_buy:
                new_sl = current_price - (base_sl * emotion_mult['sl'] * time_decay)
                new_tp = current_price + (base_tp * emotion_mult['tp'] * momentum_factor)
            else:
                new_sl = current_price + (base_sl * emotion_mult['sl'] * time_decay)
                new_tp = current_price - (base_tp * emotion_mult['tp'] * momentum_factor)
        
        return new_sl, new_tp
    
    def _get_strategy_configs(self) -> Dict[TradingStrategy, Dict[EmotionLevel, StrategyConfig]]:
        """Get configuration matrix for all strategy-emotion combinations"""
        configs = {}
        
        # Define base configurations for each strategy
        strategy_bases = {
            TradingStrategy.AGGRESSIVE: {
                'confidence_threshold': 0.72,
                'max_positions_per_symbol': 3,
                'position_size_multiplier': 1.3,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 2.0,
                'trade_cooldown_minutes': 2,  # lowered to allow more opportunities
                'risk_per_trade_percent': 1.2,   # medium risk
                'trailing_stop_distance': 20,
                'max_daily_trades': 20,  # slightly increased
                'profit_target_adjustment': 0.9
            },
            TradingStrategy.PASSIVE: {
                'confidence_threshold': 0.82,
                'max_positions_per_symbol': 2,
                'position_size_multiplier': 0.9,
                'stop_loss_multiplier': 2.0,
                'take_profit_multiplier': 3.5,
                'trade_cooldown_minutes': 18,
                'risk_per_trade_percent': 0.8,
                'trailing_stop_distance': 50,
                'max_daily_trades': 10,
                'profit_target_adjustment': 1.3
            },
            TradingStrategy.SCALPING: {
                'confidence_threshold': 0.70,
                'max_positions_per_symbol': 3,
                'position_size_multiplier': 1.4,
                'stop_loss_multiplier': 0.6,
                'take_profit_multiplier': 1.0,
                'trade_cooldown_minutes': 1,
                'risk_per_trade_percent': 0.7,
                'trailing_stop_distance': 12,
                'max_daily_trades': 40,
                'profit_target_adjustment': 0.7
            },
            TradingStrategy.SWING: {
                'confidence_threshold': 0.77,
                'max_positions_per_symbol': 3,
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 2.5,
                'take_profit_multiplier': 4.0,
                'trade_cooldown_minutes': 20,   # balance between activity & realism
                'risk_per_trade_percent': 1.0,
                'trailing_stop_distance': 80,
                'max_daily_trades': 8,
                'profit_target_adjustment': 1.7
            },
            TradingStrategy.BALANCED: {
                'confidence_threshold': 0.74,
                'max_positions_per_symbol': 3,
                'position_size_multiplier': 1.1,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 2.5,
                'trade_cooldown_minutes': 6,
                'risk_per_trade_percent': 1.0,
                'trailing_stop_distance': 25,
                'max_daily_trades': 16,
                'profit_target_adjustment': 1.0
            }
        }

        
        # Emotion adjustments
        emotion_adjustments = {
            EmotionLevel.FEARFUL: {
                'confidence_threshold': +0.05,
                'position_size_multiplier': 0.8,
                'max_daily_trades': 0.7,
                'risk_per_trade_percent': 0.8
            },
            EmotionLevel.CAUTIOUS: {
                'confidence_threshold': +0.03,
                'position_size_multiplier': 0.9,
                'max_daily_trades': 0.9,
                'risk_per_trade_percent': 0.9
            },
            EmotionLevel.STANDARD: {
                'confidence_threshold': 0.0,
                'position_size_multiplier': 1.0,
                'max_daily_trades': 1.0,
                'risk_per_trade_percent': 1.0
            },
            EmotionLevel.CONFIDENT: {
                'confidence_threshold': -0.03,
                'position_size_multiplier': 1.2,
                'max_daily_trades': 1.2,
                'risk_per_trade_percent': 1.2
            },
            EmotionLevel.GREEDY: {
                'confidence_threshold': -0.05,
                'position_size_multiplier': 1.4,
                'max_daily_trades': 1.4,
                'risk_per_trade_percent': 1.5
            }
        }

        
        # Generate all combinations
        for strategy, base_config in strategy_bases.items():
            configs[strategy] = {}
            for emotion, adjustments in emotion_adjustments.items():
                config_dict = base_config.copy()
                
                # Apply emotion adjustments
                config_dict['confidence_threshold'] += adjustments['confidence_threshold']
                config_dict['position_size_multiplier'] *= adjustments['position_size_multiplier']
                config_dict['max_daily_trades'] = int(config_dict['max_daily_trades'] * adjustments['max_daily_trades'])
                config_dict['risk_per_trade_percent'] *= adjustments['risk_per_trade_percent']
                
                # Clamp values to reasonable ranges
                config_dict['confidence_threshold'] = max(0.5, min(0.95, config_dict['confidence_threshold']))
                config_dict['max_daily_trades'] = max(1, config_dict['max_daily_trades'])
                
                configs[strategy][emotion] = StrategyConfig(**config_dict)
        
        return configs
    
    def _get_position_age_hours(self, position: Dict) -> float:
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


class IntelligentMultiSymbolTradingEngine:
    """Advanced trading engine supporting multiple symbols with different strategies"""
    
    def __init__(self):
        self.symbol_configs: Dict[str, SymbolConfig] = {}
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.models: Dict[str, Any] = {}
        self.last_data_fetch: Dict[str, float] = defaultdict(float)
        self.last_model_update: Dict[str, float] = defaultdict(float)
        self.position_manager = MultiSymbolPositionManager()
        self.signal_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.performance_tracker: Dict[str, Dict] = defaultdict(lambda: {
            'trades_today': 0,
            'wins': 0,
            'losses': 0,
            'daily_pnl': 0.0,
            'win_rate': 0.0,
            'avg_profit': 0.0
        })
        self.global_stats = {
            'total_positions': 0,
            'active_symbols': 0,
            'daily_profit': 0.0,
            'total_trades': 0
        }
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="TradingBot")
        self.lock = threading.Lock()
        
    def add_symbol(self, symbol: str, strategy: TradingStrategy, emotion: EmotionLevel,
                   max_positions: int = 3, custom_lot_size: float = None,
                   custom_timeframe: int = None) -> bool:
        """Add a symbol to trade with specific strategy and emotion"""
        try:
            symbol_config = SymbolConfig(
                symbol=symbol,
                strategy=strategy,
                emotion=emotion,
                max_positions=max_positions,
                custom_lot_size=custom_lot_size,
                custom_timeframe=custom_timeframe or state.timeframe
            )
            
            self.symbol_configs[symbol] = symbol_config
            
            # Initialize data and model for the symbol
            self._initialize_symbol_data(symbol)
            
            logger.info(f"Added symbol {symbol} with strategy {strategy.value} and emotion {emotion.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {str(e)}")
            return False
    
    def update_symbol_emotion(self, symbol: str, new_emotion: EmotionLevel):
        """Update the emotion level for a specific symbol"""
        if symbol in self.symbol_configs:
            old_emotion = self.symbol_configs[symbol].emotion
            self.symbol_configs[symbol].emotion = new_emotion
            logger.info(f"Updated {symbol} emotion: {old_emotion.value} -> {new_emotion.value}")
        else:
            logger.warning(f"Symbol {symbol} not found in configurations")
    
    def update_symbol_strategy(self, symbol: str, new_strategy: TradingStrategy):
        """Update the strategy for a specific symbol"""
        if symbol in self.symbol_configs:
            old_strategy = self.symbol_configs[symbol].strategy
            self.symbol_configs[symbol].strategy = new_strategy
            logger.info(f"Updated {symbol} strategy: {old_strategy.value} -> {new_strategy.value}")
        else:
            logger.warning(f"Symbol {symbol} not found in configurations")
    
    def enable_symbol(self, symbol: str, enabled: bool = True):
        """Enable or disable trading for a specific symbol"""
        if symbol in self.symbol_configs:
            self.symbol_configs[symbol].enabled = enabled
            status = "enabled" if enabled else "disabled"
            logger.info(f"Symbol {symbol} {status}")
        else:
            logger.warning(f"Symbol {symbol} not found in configurations")
    
    def _initialize_symbol_data(self, symbol: str):
        """Initialize data and model for a new symbol"""
        try:
            config = self.symbol_configs[symbol]
            timeframe = config.custom_timeframe or state.timeframe
            
            # Fetch historical data
            df = fetch_historical_data(symbol, timeframe, state.num_bars)
            if df is not None:
                self.data_cache[symbol] = df
                self.last_data_fetch[symbol] = time.time()
                
                # Train initial model
                df_features = fe.engineer_features(df)
                if df_features is not None and 'Target' in df_features.columns:
                    X = df_features.drop(columns=['Target', 'Future_Return'], errors='ignore')
                    y = df_features['Target']
                    model = self._train_model(X, y, symbol)
                    if model is not None:
                        self.models[symbol] = model
                        self.last_model_update[symbol] = time.time()
                        
                logger.info(f"Initialized data and model for {symbol}")
                return True
            else:
                logger.error(f"Failed to fetch initial data for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing symbol data for {symbol}: {str(e)}")
            return False
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, symbol: str) -> Optional[Any]:
        """Train model for specific symbol"""
        try:
            if len(np.unique(y)) < 2:
                logger.warning(f"Insufficient class diversity for {symbol}")
                return None
            
            # Get strategy config for model parameters
            config = self.symbol_configs[symbol]
            strategy_configs = self.position_manager._get_strategy_configs()
            strategy_config = strategy_configs[config.strategy][config.emotion]
            
            # Adjust model complexity based on strategy
            if config.strategy == TradingStrategy.SCALPING:
                # Simpler, faster model for scalping
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=5,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1
                )
            elif config.strategy == TradingStrategy.SWING:
                # More complex model for swing trading
                model = GradientBoostingClassifier(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.05,
                    min_samples_split=15,
                    min_samples_leaf=8,
                    random_state=42
                )
            else:
                # Balanced model for other strategies
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Train with cross-validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            
            logger.info(f"Model trained for {symbol}: accuracy = {score:.3f}")
            return model
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {str(e)}")
            return None
    
    def _should_fetch_new_data(self, symbol: str) -> bool:
        """Check if we should fetch new data for symbol"""
        current_time = time.time()
        last_fetch = self.last_data_fetch.get(symbol, 0)
        
        config = self.symbol_configs.get(symbol)
        if not config:
            return False
        
        # Fetch interval based on strategy
        if config.strategy == TradingStrategy.SCALPING:
            interval = 60  # 1 minute
        elif config.strategy == TradingStrategy.AGGRESSIVE:
            interval = 300  # 5 minutes
        elif config.strategy == TradingStrategy.SWING:
            interval = 300  # 30 minutes
        else:
            interval = 600  # 10 minutes default
        
        return current_time - last_fetch > interval
    
    def _calculate_dynamic_confidence_threshold(self, symbol: str) -> float:
        """Calculate dynamic confidence threshold for symbol"""
        config = self.symbol_configs.get(symbol)
        if not config:
            return 0.75
        
        strategy_configs = self.position_manager._get_strategy_configs()
        base_threshold = strategy_configs[config.strategy][config.emotion].confidence_threshold
        
        # Adjust based on performance
        performance = self.performance_tracker[symbol]
        if performance['trades_today'] > 0:
            win_rate = performance['win_rate']
            if win_rate < 0.3:  # Poor performance
                base_threshold += 0.15
            elif win_rate > 0.8:  # Excellent performance
                base_threshold -= 0.05
        
        # Adjust based on daily P&L
        if performance['daily_pnl'] < -50:  # Significant losses
            base_threshold += 0.20
        elif performance['daily_pnl'] > 100:  # Good profits
            base_threshold -= 0.03
        
        # Clamp to a narrower, more permissive band
        return max(0.60, min(0.85, base_threshold))
    
    def _generate_signal_for_symbol(self, symbol: str) -> Tuple[Optional[int], float, str]:
        """Generate trading signal for specific symbol"""
        try:
            config = self.symbol_configs.get(symbol)
            if not config or not config.enabled:
                return None, 0.0, "Symbol disabled"
            
            model = self.models.get(symbol)
            df = self.data_cache.get(symbol)
            
            if model is None or df is None or len(df) < 50:
                return None, 0.0, "Insufficient data or model"
            
            # Generate features
            df_features = fe.engineer_features(df)
            if df_features is None or 'Target' not in df_features.columns:
                return None, 0.0, "Feature engineering failed"
            
            X = df_features.drop(columns=['Target', 'Future_Return'], errors='ignore')
            latest_features = X.iloc[-1:].fillna(0)
            
            # Get prediction
            prediction = model.predict(latest_features)[0]
            probabilities = model.predict_proba(latest_features)[0]
            confidence = probabilities.max()
            
            # Check dynamic threshold
            dynamic_threshold = self._calculate_dynamic_confidence_threshold(symbol)
            
            # Signal filtering
            reasons = []
            
            if confidence < dynamic_threshold:
                reasons.append(f"Low confidence: {confidence:.3f} < {dynamic_threshold:.3f}")
            
            # Skip neutral/hold predictions entirely
            if prediction == 0:
                return None, confidence, "Neutral prediction"

            # Check signal consistency (require at least 2 same recent) with high-confidence bypass
            self.signal_history[symbol].append((prediction, confidence))
            recent = list(self.signal_history[symbol])[-3:]
            recent_signals = [s[0] for s in recent]
            same_last_two = len(recent_signals) >= 2 and recent_signals[-1] == recent_signals[-2]
            high_confidence = confidence >= (dynamic_threshold + 0.05)
            if not same_last_two and not high_confidence:
                reasons.append("Inconsistent recent signals")
            
            # Check cooldown
            strategy_configs = self.position_manager._get_strategy_configs()
            cooldown_minutes = strategy_configs[config.strategy][config.emotion].trade_cooldown_minutes
            if time.time() - config.last_trade_time < cooldown_minutes * 60:
                if not high_confidence:
                    reasons.append("Trade cooldown active")
            
            # Check daily limits
            max_daily = strategy_configs[config.strategy][config.emotion].max_daily_trades
            if config.daily_trades >= max_daily:
                if not high_confidence:
                    reasons.append("Daily trade limit reached")
            
            # Check max positions
            current_positions = len(self._get_symbol_positions(symbol))
            if current_positions >= config.max_positions:
                reasons.append("Max positions reached")
            
            if reasons:
                return None, confidence, "; ".join(reasons)
            
            return prediction, confidence, "Signal passed all filters"
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            return None, 0.0, f"Signal generation error: {str(e)}"
    
    def _calculate_position_size(self, symbol: str, confidence: float) -> float:
        """Calculate position size for symbol based on strategy and emotion"""
        try:
            config = self.symbol_configs.get(symbol)
            if not config:
                return 0.01
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0.01
            
            # Base size
            base_size = config.custom_lot_size or state.lot_size
            
            # Get strategy configuration
            strategy_configs = self.position_manager._get_strategy_configs()
            strategy_config = strategy_configs[config.strategy][config.emotion]
            
            # Calculate size based on strategy and confidence
            confidence_multiplier = min(confidence / 0.75, 1.5)
            strategy_multiplier = strategy_config.position_size_multiplier
            
            # Risk-based sizing
            account_info = mt5.account_info()
            if account_info:
                risk_amount = account_info.balance * (strategy_config.risk_per_trade_percent / 100)
                risk_based_size = risk_amount / (base_size * 1000)  # Approximate calculation
                base_size = min(base_size, risk_based_size)
            
            # Performance-based adjustment
            performance = self.performance_tracker[symbol]
            performance_multiplier = 1.0
            if performance['trades_today'] > 0:
                win_rate = performance['win_rate']
                if win_rate > 0.7:
                    performance_multiplier = 1.2
                elif win_rate < 0.4:
                    performance_multiplier = 0.7
            
            # Calculate final size
            raw_size = base_size * confidence_multiplier * strategy_multiplier * performance_multiplier
            
            # Validate against broker limits
            volume_step = symbol_info.volume_step or 0.01
            volume_min = symbol_info.volume_min or 0.01
            volume_max = symbol_info.volume_max or 100.0
            
            final_size = round(raw_size / volume_step) * volume_step
            final_size = max(volume_min, min(final_size, volume_max))
            
            logger.debug(f"{symbol} position size: {final_size:.2f} (confidence: {confidence:.3f})")
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {str(e)}")
            return 0.01
    
    def _get_symbol_positions(self, symbol: str) -> List[Dict]:
        """Get all open positions for a symbol"""
        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                return []
            
            return [
                {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': pos.type,
                    'volume': pos.volume,
                    'open_price': pos.price_open,
                    'current_price': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'open_time': pos.time,
                    'magic': pos.magic
                }
                for pos in positions
            ]
            
        except Exception as e:
            logger.error(f"Error getting positions for {symbol}: {str(e)}")
            return []
    
    def _execute_trade_for_symbol(self, symbol: str, signal: int, confidence: float) -> bool:
        """Execute trade for specific symbol"""
        try:
            config = self.symbol_configs.get(symbol)
            if not config:
                return False
            
            action = "buy" if signal == 1 else "sell"
            lot_size = self._calculate_position_size(symbol, confidence)
            
            # Execute trade using the existing execute_trade function
            success = execute_trade(symbol, action, lot_size)
            
            if success:
                # Update tracking
                config.last_trade_time = time.time()
                config.daily_trades += 1
                self.performance_tracker[symbol]['trades_today'] += 1
                self.global_stats['total_trades'] += 1
                
                logger.info(f"Executed {action.upper()} for {symbol}: {lot_size} lots "
                           f"(confidence: {confidence:.3f}, strategy: {config.strategy.value}, "
                           f"emotion: {config.emotion.value})")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {str(e)}")
            return False
    
    def _manage_symbol_positions(self, symbol: str):
        """Manage positions for specific symbol"""
        try:
            positions = self._get_symbol_positions(symbol)
            if not positions:
                return
            
            config = self.symbol_configs.get(symbol)
            if not config:
                return
            
            df = self.data_cache.get(symbol)
            if df is None:
                return
            
            # Calculate momentum and volatility
            momentum = self.position_manager.analyze_symbol_momentum(symbol, df)
            volatility_deque = self.position_manager.symbol_volatility[symbol]
            volatility = np.mean(volatility_deque) if volatility_deque else 0.01
            
            for position in positions:
                try:
                    # Get current price
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        continue
                    
                    position_type = position['type']
                    current_price = tick.bid if position_type == mt5.ORDER_TYPE_BUY else tick.ask
                    position['current_price'] = current_price
                    
                    # Calculate profit percentage
                    open_price = position['open_price']
                    if position_type == mt5.ORDER_TYPE_BUY:
                        profit_pct = ((current_price - open_price) / open_price) * 100
                    else:
                        profit_pct = ((open_price - current_price) / open_price) * 100
                    
                    position['profit_pct'] = profit_pct
                    
                    # Check for emergency closure
                    if profit_pct <= self.position_manager.profit_zones['emergency_exit']:
                        position['should_close'] = True
                        position['close_reason'] = f'Emergency exit at {profit_pct:.2f}%'
                        self._close_position(position, symbol)
                        continue

                    # Proactive profit-taking rules
                    # Emotion-based aggressiveness: more conservative emotions take profit earlier
                    emotion = config.emotion
                    strategy = config.strategy
                    quick = self.position_manager.profit_zones['quick_profit']
                    target = self.position_manager.profit_zones['target_profit']
                    superp = self.position_manager.profit_zones['super_profit']

                    # Adjust thresholds by emotion and momentum
                    emotion_factor = {
                        EmotionLevel.FEARFUL: 0.7,
                        EmotionLevel.CAUTIOUS: 0.85,
                        EmotionLevel.STANDARD: 1.0,
                        EmotionLevel.CONFIDENT: 1.15,
                        EmotionLevel.GREEDY: 1.3
                    }[emotion]

                    # Determine side for momentum alignment
                    is_buy = (position_type == mt5.ORDER_TYPE_BUY)
                    momentum_adj = 0.9 if (momentum < 0 and is_buy) else (0.9 if (momentum > 0 and not is_buy) else 1.0)

                    quick_thr = quick * emotion_factor * momentum_adj
                    target_thr = target * emotion_factor
                    super_thr = superp * emotion_factor

                    # Time in trade factor – take profits earlier if trade lingers
                    hours_open = self.position_manager._get_position_age_hours(position)
                    linger_factor = 0.9 if hours_open > 6 else 1.0

                    # Define partial steps (simple full close first iteration)
                    take_quick = profit_pct >= (quick_thr * linger_factor)
                    take_target = profit_pct >= (target_thr * linger_factor)
                    take_super = profit_pct >= (super_thr * linger_factor)

                    # Strategy nuance: scalping exits earlier, swing lets run more
                    if strategy == TradingStrategy.SCALPING and take_quick:
                        position['should_close'] = True
                        position['close_reason'] = f'Quick profit lock ({profit_pct:.2f}%)'
                        self._close_position(position, symbol)
                        continue

                    if strategy in (TradingStrategy.AGGRESSIVE, TradingStrategy.BALANCED):
                        if take_target:
                            # Close at target; aggressive might wait for super if momentum strong
                            if strategy == TradingStrategy.AGGRESSIVE and momentum > 0.2 and profit_pct < super_thr:
                                pass  # let it run, SL/TP adjustment below
                            else:
                                position['should_close'] = True
                                position['close_reason'] = f'Target profit lock ({profit_pct:.2f}%)'
                                self._close_position(position, symbol)
                                continue

                    if strategy == TradingStrategy.SWING:
                        if take_super and momentum <= 0:
                            position['should_close'] = True
                            position['close_reason'] = f'Super profit realized ({profit_pct:.2f}%)'
                            self._close_position(position, symbol)
                            continue
                    
                    # Calculate new SL/TP based on strategy
                    new_sl, new_tp, _ = self.position_manager.calculate_strategy_based_sltp(
                        position, config, current_price, volatility, momentum
                    )
                    
                    if new_sl is not None or new_tp is not None:
                        should_modify = self._should_modify_position(position, new_sl, new_tp)
                        if should_modify:
                            self._modify_position_sltp(position, new_sl, new_tp, 
                                                     f"{config.strategy.value}-{config.emotion.value} adjustment")
                    
                except Exception as e:
                    logger.error(f"Error managing position {position.get('ticket', 'N/A')}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error managing positions for {symbol}: {str(e)}")
    
    def _should_modify_position(self, position: Dict, new_sl: Optional[float], new_tp: Optional[float]) -> bool:
        """Check if position should be modified"""
        try:
            current_sl = position.get('sl', 0.0)
            current_tp = position.get('tp', 0.0)
            
            # Always modify if no SL/TP set
            if (new_sl is not None and current_sl == 0.0) or (new_tp is not None and current_tp == 0.0):
                return True
            
            # Check minimum movement
            symbol_info = mt5.symbol_info(position['symbol'])
            if symbol_info:
                min_movement = 5 * symbol_info.point  # 5 pips minimum
            else:
                min_movement = 0.00005  # Default
            
            # Check SL movement
            if new_sl is not None and current_sl != 0.0:
                sl_movement = abs(new_sl - current_sl)
                if sl_movement > min_movement:
                    return True
            
            # Check TP movement
            if new_tp is not None and current_tp != 0.0:
                tp_movement = abs(new_tp - current_tp)
                if tp_movement > min_movement:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking position modification: {str(e)}")
            return False
    
    def _modify_position_sltp(self, position: Dict, new_sl: Optional[float], 
                             new_tp: Optional[float], reason: str) -> bool:
        """Modify position SL/TP"""
        try:
            ticket = position['ticket']
            symbol = position['symbol']
            
            # Validate levels
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
            
            min_stop_level = symbol_info.trade_stops_level * symbol_info.point
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False
            
            # Ensure symbol selected and tick available
            try:
                if not symbol_info.visible:
                    mt5.symbol_select(symbol, True)
            except Exception:
                pass
            # Validate stop distances
            position_type = position['type']
            current_price = tick.bid if position_type == mt5.ORDER_TYPE_BUY else tick.ask
            
            spread_points = symbol_info.spread or 0
            freeze_level = getattr(symbol_info, 'freeze_level', 0) or 0
            safety = max(min_stop_level, freeze_level * symbol_info.point, spread_points * symbol_info.point * 3)

            if position_type == mt5.ORDER_TYPE_BUY:
                if new_sl is not None and current_price - new_sl < safety:
                    new_sl = current_price - safety
                if new_tp is not None and new_tp - current_price < safety:
                    new_tp = current_price + safety
            else:
                if new_sl is not None and new_sl - current_price < safety:
                    new_sl = current_price + safety
                if new_tp is not None and current_price - new_tp < safety:
                    new_tp = current_price - safety
            
            # Send modification request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": int(ticket),
                "sl": float(new_sl) if new_sl is not None else position.get('sl', 0.0),
                "tp": float(new_tp) if new_tp is not None else position.get('tp', 0.0)
            }
            
            result = mt5.order_send(request)
            
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Modified {symbol} position {ticket}: "
                    f"SL={'unchanged' if new_sl is None else f'{new_sl:.5f}'}, "
                    f"TP={'unchanged' if new_tp is None else f'{new_tp:.5f}'} - {reason}")
                return True
            else:
                logger.warning(f"Failed to modify {symbol} position {ticket}: {result.comment if result else 'Unknown error'}")
                return False
                
        except Exception as e:
            logger.error(f"Error modifying position: {str(e)}")
            return False
    
    def _close_position(self, position: Dict, symbol: str) -> bool:
        """Close position with performance tracking"""
        try:
            success = close_position(position)  # Use existing close_position function
            
            if success:
                # Update performance tracking
                config = self.symbol_configs.get(symbol)
                if config:
                    profit_pct = position.get('profit_pct', 0)
                    profit_amount = position.get('profit', 0)
                    
                    performance = self.performance_tracker[symbol]
                    if profit_pct > 0:
                        performance['wins'] += 1
                    else:
                        performance['losses'] += 1
                    
                    performance['daily_pnl'] += profit_amount
                    config.daily_pnl += profit_amount
                    
                    # Update win rate
                    total_trades = performance['wins'] + performance['losses']
                    if total_trades > 0:
                        performance['win_rate'] = performance['wins'] / total_trades
                    
                    # Update global stats
                    self.global_stats['daily_profit'] += profit_amount
                    self.global_stats['total_positions'] -= 1
                    
                    logger.info(f"Closed {symbol} position: {profit_pct:.2f}% profit, "
                               f"Daily P&L: ${performance['daily_pnl']:.2f}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}")
            return False
    
    def get_status_report(self) -> Dict:
        """Get comprehensive status report"""
        try:
            report = {
                'global_stats': self.global_stats.copy(),
                'symbol_performance': {},
                'active_positions': {},
                'strategy_distribution': defaultdict(int),
                'emotion_distribution': defaultdict(int)
            }
            
            # Update global stats
            total_positions = 0
            active_symbols = 0
            
            for symbol, config in self.symbol_configs.items():
                if config.enabled:
                    active_symbols += 1
                    positions = self._get_symbol_positions(symbol)
                    total_positions += len(positions)
                    
                    # Performance data
                    performance = self.performance_tracker[symbol]
                    report['symbol_performance'][symbol] = {
                        'strategy': config.strategy.value,
                        'emotion': config.emotion.value,
                        'enabled': config.enabled,
                        'daily_trades': config.daily_trades,
                        'max_positions': config.max_positions,
                        'current_positions': len(positions),
                        'daily_pnl': performance['daily_pnl'],
                        'win_rate': performance['win_rate'],
                        'wins': performance['wins'],
                        'losses': performance['losses']
                    }
                    
                    # Active positions
                    if positions:
                        report['active_positions'][symbol] = [
                            {
                                'ticket': pos['ticket'],
                                'type': 'BUY' if pos['type'] == mt5.ORDER_TYPE_BUY else 'SELL',
                                'volume': pos['volume'],
                                'profit': pos['profit'],
                                'profit_pct': pos.get('profit_pct', 0)
                            }
                            for pos in positions
                        ]
                    
                    # Distribution counts
                    report['strategy_distribution'][config.strategy.value] += 1
                    report['emotion_distribution'][config.emotion.value] += 1
            
            report['global_stats']['total_positions'] = total_positions
            report['global_stats']['active_symbols'] = active_symbols
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating status report: {str(e)}")
            return {'error': str(e)}
    
    async def run_intelligent_trading_loop(self):
        """Main intelligent trading loop with multi-symbol support"""
        logger.info("Starting intelligent multi-symbol trading loop...")
        
        try:
            # Initialize MT5
            if not check_mt5_connection():
                if hasattr(state, 'mt5_login') and state.mt5_login:
                    initialize_mt5(state.mt5_login, state.mt5_password, state.mt5_server)
                else:
                    logger.error("MT5 credentials not available")
                    return
            
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
                    
                    # Process all enabled symbols concurrently
                    active_symbols = [symbol for symbol, config in self.symbol_configs.items() if config.enabled]
                    
                    if not active_symbols:
                        logger.warning("No active symbols configured")
                        await asyncio.sleep(30)
                        continue
                    
                    # Data fetching phase
                    data_tasks = []
                    for symbol in active_symbols:
                        if self._should_fetch_new_data(symbol):
                            task = self.executor.submit(self._fetch_and_update_data, symbol)
                            data_tasks.append((symbol, task))
                    
                    # Process data updates
                    for symbol, task in data_tasks:
                        try:
                            task.result(timeout=30)  # 30 second timeout
                        except Exception as e:
                            logger.error(f"Data fetch timeout/error for {symbol}: {str(e)}")
                    
                    # Signal generation and trading phase
                    trading_tasks = []
                    for symbol in active_symbols:
                        task = self.executor.submit(self._process_symbol_trading, symbol)
                        trading_tasks.append((symbol, task))
                    
                    # Process trading signals
                    for symbol, task in trading_tasks:
                        try:
                            task.result(timeout=15)  # 15 second timeout
                        except Exception as e:
                            logger.error(f"Trading processing timeout/error for {symbol}: {str(e)}")
                    
                    # Position management phase
                    if iteration_count % 3 == 0:  # Every 3rd iteration
                        position_tasks = []
                        for symbol in active_symbols:
                            task = self.executor.submit(self._manage_symbol_positions, symbol)
                            position_tasks.append((symbol, task))
                        
                        # Process position management
                        for symbol, task in position_tasks:
                            try:
                                task.result(timeout=20)  # 20 second timeout
                            except Exception as e:
                                logger.error(f"Position management timeout/error for {symbol}: {str(e)}")
                    
                    # Status reporting
                    if current_time - last_status_log > 300:  # Every 5 minutes
                        report = self.get_status_report()
                        logger.info(f"Multi-Symbol Status: Active={report['global_stats']['active_symbols']}, "
                                   f"Positions={report['global_stats']['total_positions']}, "
                                   f"Profit=${report['global_stats']['daily_profit']:.2f}")
                        last_status_log = current_time
                    
                    # Adaptive sleep based on market activity
                    sleep_time = 5  # Base 5 seconds
                    if len(active_symbols) > 5:
                        sleep_time = 3  # Faster for many symbols
                    elif any(config.strategy == TradingStrategy.SCALPING for config in self.symbol_configs.values()):
                        sleep_time = 2  # Faster for scalping
                    
                    await asyncio.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop iteration: {str(e)}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"Fatal error in intelligent trading loop: {str(e)}")
            logger.error(traceback.format_exc())
            state.running = False
    
    def _fetch_and_update_data(self, symbol: str):
        """Fetch and update data for symbol (thread-safe)"""
        try:
            config = self.symbol_configs.get(symbol)
            if not config:
                return
            
            timeframe = config.custom_timeframe or state.timeframe
            df = fetch_historical_data(symbol, timeframe, state.num_bars)
            
            if df is not None:
                with self.lock:
                    self.data_cache[symbol] = df
                    self.last_data_fetch[symbol] = time.time()
                
                # Update model periodically (every 4 hours)
                if time.time() - self.last_model_update.get(symbol, 0) > 14400:
                    df_features = fe.engineer_features(df)
                    if df_features is not None and 'Target' in df_features.columns:
                        X = df_features.drop(columns=['Target', 'Future_Return'], errors='ignore')
                        y = df_features['Target']
                        new_model = self._train_model(X, y, symbol)
                        if new_model is not None:
                            with self.lock:
                                self.models[symbol] = new_model
                                self.last_model_update[symbol] = time.time()
                
                logger.debug(f"Updated data for {symbol}")
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
    
    def _process_symbol_trading(self, symbol: str):
        """Process trading logic for symbol (thread-safe)"""
        try:
            # Generate signal
            signal, confidence, reason = self._generate_signal_for_symbol(symbol)
            
            if signal is not None:
                logger.info(f"{symbol} signal: {signal} (confidence: {confidence:.3f})")
                success = self._execute_trade_for_symbol(symbol, signal, confidence)
                if not success:
                    logger.debug(f"{symbol} trade execution failed or skipped")
            else:
                logger.debug(f"{symbol} no signal: {reason}")
                
        except Exception as e:
            logger.error(f"Error processing trading for {symbol}: {str(e)}")

    # Usage example functions for the new system
# def setup_multi_symbol_trading():
#     """Example setup for multi-symbol trading with different strategies"""
#     try:
#         # Example: Add multiple symbols with different strategies and emotions
        
#         # Aggressive scalping setup
#         intelligent_engine.add_symbol(
#             symbol="EURUSD",
#             strategy=TradingStrategy.SCALPING,
#             emotion=EmotionLevel.CONFIDENT,
#             max_positions=5,
#             custom_lot_size=0.01
#         )
        
#         # Conservative swing trading
#         intelligent_engine.add_symbol(
#             symbol="GBPUSD", 
#             strategy=TradingStrategy.SWING,
#             emotion=EmotionLevel.CAUTIOUS,
#             max_positions=2,
#             custom_lot_size=0.02
#         )
        
#         # Balanced approach
#         intelligent_engine.add_symbol(
#             symbol="USDJPY",
#             strategy=TradingStrategy.BALANCED,
#             emotion=EmotionLevel.STANDARD,
#             max_positions=3,
#             custom_lot_size=0.015
#         )
        
#         # Aggressive day trading
#         intelligent_engine.add_symbol(
#             symbol="AUDUSD",
#             strategy=TradingStrategy.AGGRESSIVE,
#             emotion=EmotionLevel.GREEDY,
#             max_positions=4,
#             custom_lot_size=0.01
#         )
        
#         # Passive position trading
#         intelligent_engine.add_symbol(
#             symbol="USDCAD",
#             strategy=TradingStrategy.PASSIVE,
#             emotion=EmotionLevel.FEARFUL,
#             max_positions=2,
#             custom_lot_size=0.01
#         )
        
#         logger.info("Multi-symbol trading setup completed")
#         return True
        
#     except Exception as e:
#         logger.error(f"Error setting up multi-symbol trading: {str(e)}")
#         return False
    # def setup_multi_symbol_trading(self):
    #     """Setup multi-symbol trading with different strategies"""
    #     try:
    #         self.add_symbol("EURUSD", strategy=TradingStrategy.SCALPING, emotion=EmotionLevel.CONFIDENT, max_positions=5, custom_lot_size=0.01)
    #         self.add_symbol("GBPUSD", strategy=TradingStrategy.SWING, emotion=EmotionLevel.CAUTIOUS, max_positions=2, custom_lot_size=0.02)
    #         self.add_symbol("USDJPY", strategy=TradingStrategy.BALANCED, emotion=EmotionLevel.STANDARD, max_positions=3, custom_lot_size=0.015)
    #         self.add_symbol("AUDUSD", strategy=TradingStrategy.AGGRESSIVE, emotion=EmotionLevel.GREEDY, max_positions=4, custom_lot_size=0.01)
    #         self.add_symbol("USDCAD", strategy=TradingStrategy.PASSIVE, emotion=EmotionLevel.FEARFUL, max_positions=2, custom_lot_size=0.01)

    #         logger.info("Multi-symbol trading setup completed")
    #     except Exception as e:
    #         logger.error(f"Error setting up multi-symbol trading: {str(e)}")


    def setup_multi_symbol_trading(self, max_symbols: int = 10, retry_count: int = 3) -> bool:
        """Robust automatic symbol configuration with proper symbol analysis"""
        def fallback_setup():
            fallback_symbols = [
                ("EURUSD", TradingStrategy.SCALPING, EmotionLevel.CONFIDENT, 3, 0.01),
                ("GBPUSD", TradingStrategy.SWING, EmotionLevel.CAUTIOUS, 2, 0.02),
                ("USDJPY", TradingStrategy.BALANCED, EmotionLevel.STANDARD, 2, 0.015),
                ("EURGBP", TradingStrategy.SWING, EmotionLevel.CAUTIOUS, 2, 0.01),
                ("AUDNZD", TradingStrategy.BALANCED, EmotionLevel.STANDARD, 2, 0.01),
                ("GBPJPY", TradingStrategy.AGGRESSIVE, EmotionLevel.GREEDY, 2, 0.005),
                ("EURJPY", TradingStrategy.SCALPING, EmotionLevel.CONFIDENT, 3, 0.01),
            ]
            success_count = 0
            for symbol, strategy, emotion, max_pos, lot_size in fallback_symbols:
                if self.add_symbol(symbol, strategy, emotion, max_pos, lot_size):
                    success_count += 1
            return success_count > 0

        for attempt in range(1, retry_count + 1):
            try:
                if not mt5.initialize():
                    logger.error("MT5 not initialized, retrying...")
                    continue

                # Initialize symbol analyzer and allocator
                analyzer = SymbolAnalyzer()
                allocator = StrategyAllocator()
                
                # Get properly analyzed symbols
                tradable_symbols = analyzer.analyze_all_symbols()
                
                if not tradable_symbols:
                    logger.warning(f"Attempt {attempt}: No tradable symbols found")
                    continue

                # Select subset if too many
                selected_symbols = (tradable_symbols[:max_symbols] 
                                if len(tradable_symbols) <= max_symbols 
                                else random.sample(tradable_symbols, max_symbols))

                # Configure with proper strategy allocation
                success_count = 0
                for symbol in selected_symbols:
                    profile = analyzer.symbol_profiles.get(symbol)
                    if not profile:
                        continue
                        
                    # Get strategy based on actual market analysis
                    strategy_config = allocator.get_strategy_for_symbol(symbol, profile)
                    
                    if self.add_symbol(
                        symbol=symbol,
                        strategy=TradingStrategy[strategy_config['strategy']],
                        emotion=EmotionLevel[strategy_config['emotion']],
                        max_positions=strategy_config['max_positions'],
                        custom_lot_size=strategy_config['lot_size']
                    ):
                        success_count += 1

                if success_count > 0:
                    logger.info(f"Configured {success_count} symbols successfully")
                    return True

            except Exception as e:
                logger.error(f"Setup attempt {attempt} failed: {str(e)}")
                logger.error(traceback.format_exc())

        # Fallback if all attempts fail
        logger.warning("Auto setup failed, using fallback configuration")
        return fallback_setup()

    def connect_autonomous_manager(self):
        """Inject autonomous management"""
        from .symbol_manager import AutonomousSymbolManager
        self.autonomous_manager = AutonomousSymbolManager(self)
        return self.autonomous_manager

# Global instance
intelligent_engine = IntelligentMultiSymbolTradingEngine()


# Enhanced execute_trade function with symbol validation
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

        # Calculate volatility for SL/TP
        df = fetch_historical_data(symbol, mt5.TIMEFRAME_M5, 200)
        if df is None or df.empty:
            logger.error("Failed to fetch data for volatility calculation")
            return False
        
        # More conservative volatility calculation
        recent_prices = df['close'].tail(50)
        volatility = recent_prices.std() * 2  # Reduced multiplier
        logger.info(f"Calculated volatility: {volatility}")

        # Ensure symbol is selected and we have fresh tick
        try:
            if not symbol_info.visible:
                mt5.symbol_select(symbol, True)
        except Exception:
            pass
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick data for {symbol}")
            return False

        # Broker min distances and freeze levels
        min_stop_points = symbol_info.trade_stops_level or 0
        freeze_points = getattr(symbol_info, 'freeze_level', 0) or 0
        point_value = symbol_info.point
        spread_points = int((symbol_info.spread or 0))
        # Base min distance in points: at least 10 points or 3x spread
        safety_points = max(10, min_stop_points, freeze_points, spread_points * 3)
        min_stop_distance = safety_points * point_value
        
        logger.info(f"Minimum stop level: {min_stop_points} points, freeze: {freeze_points} points, using min distance {safety_points} points ({min_stop_distance} price distance)")

        # JPY handling: wider min distances due to pip scale
        is_jpy = symbol.endswith("JPY")
        jpy_multiplier = 2.0 if is_jpy else 1.0

        # Set order parameters with more reasonable SL/TP
        if action == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            # Conservative SL/TP calculation with safety floor
            sl_distance = max(volatility, min_stop_distance * 2 * jpy_multiplier)
            tp_distance = max(sl_distance * 2, min_stop_distance * 3 * jpy_multiplier)
            sl = price - sl_distance
            tp = price + tp_distance
        elif action == "sell":
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            sl_distance = max(volatility, min_stop_distance * 2 * jpy_multiplier)
            tp_distance = max(sl_distance * 2, min_stop_distance * 3 * jpy_multiplier)
            sl = price + sl_distance
            tp = price - tp_distance
        else:
            logger.error("Invalid action, skipping trade")
            return False

        # Validate parameters
        if None in (sl, tp, price, order_type):
            logger.error("Missing critical trade parameters!")
            return False

        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": max(20, spread_points * 2),
            "magic": 234000,
            "comment": "Smart multi-symbol bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        logger.info(f"Sending trade request: {request}")
        result = mt5.order_send(request)
        
        # Retry with IOC and wider distances if invalid stops or filling mode issue
        if result is None or result.retcode not in (mt5.TRADE_RETCODE_DONE,):
            err_code = getattr(result, 'retcode', None)
            err_comment = getattr(result, 'comment', 'No result')
            if err_code in (mt5.TRADE_RETCODE_INVALID_STOPS, 10016, mt5.TRADE_RETCODE_INVALID_PRICE):
                # Widen distances
                sl_buffer = min_stop_distance * (2.0 * jpy_multiplier)
                tp_buffer = min_stop_distance * (3.0 * jpy_multiplier)
                if order_type == mt5.ORDER_TYPE_BUY:
                    sl = min(sl, price - sl_buffer)
                    tp = max(tp, price + tp_buffer)
                else:
                    sl = max(sl, price + sl_buffer)
                    tp = min(tp, price - tp_buffer)
                request.update({
                    "sl": float(sl),
                    "tp": float(tp),
                    "type_filling": mt5.ORDER_FILLING_IOC,
                    "deviation": max(request["deviation"], spread_points * 3)
                })
                logger.info(f"Retrying with IOC and widened SL/TP: {request}")
                result = mt5.order_send(request)

        if result is None:
            logger.error("Order send returned None")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to execute {action} order: {result.comment}, Retcode: {result.retcode}")
            return False
            
        logger.info(f"{action.upper()} order executed: Price={price:.5f}, SL={sl:.5f}, TP={tp:.5f}, Lot Size={lot_size}")
        return True

    except Exception as e:
        logger.error(f"Exception in execute_trade: {str(e)}")
        logger.error(traceback.format_exc())
        state.error_count += 1
        state.last_error = str(e)
        return False


def close_position(position, max_retries=3):
    """Enhanced position closing with comprehensive error handling and retries"""
    try:
        # Validate essential position data
        required_fields = ['symbol', 'volume', 'ticket']
        missing_fields = [field for field in required_fields if field not in position]
        
        if missing_fields:
            logger.error(f"Invalid position data: missing fields {missing_fields}")
            return False

        symbol = position['symbol']
        volume = position['volume']
        ticket = position['ticket']
        
        # Verify position still exists
        current_positions = mt5.positions_get(ticket=ticket)
        if not current_positions:
            logger.info(f"Position {ticket} already closed")
            return True
        
        # Get position type from current MT5 data (most reliable)
        position_type = current_positions[0].type
        
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
        
        # Validate volume
        if not (symbol_info.volume_min <= volume <= symbol_info.volume_max):
            logger.error(f"Volume {volume} outside allowed range [{symbol_info.volume_min}, {symbol_info.volume_max}]")
            return False

        # Determine close order type and price
        if position_type == mt5.ORDER_TYPE_BUY:
            close_order_type = mt5.ORDER_TYPE_SELL
            close_price = tick.bid
        elif position_type == mt5.ORDER_TYPE_SELL:
            close_order_type = mt5.ORDER_TYPE_BUY
            close_price = tick.ask
        else:
            logger.error(f"Unknown position type: {position_type}")
            return False
        
        # Validate price
        if close_price <= 0:
            logger.error(f"Invalid close price: {close_price}")
            return False

        # Prepare close request
        deviation = min(symbol_info.spread * 3, 50)  # Dynamic deviation based on spread
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": close_order_type,
            "position": int(ticket),
            "price": float(close_price),
            "deviation": deviation,
            "magic": position.get('magic', 234000),
            "comment": f"Auto-close: {position.get('close_reason', 'Strategy-based')}",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        logger.info(f"Closing position {ticket}: {symbol} vol:{volume} at {close_price}")

        # Send close order with retries
        for attempt in range(max_retries):
            result = mt5.order_send(request)
            
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Successfully closed position {ticket} - {position.get('close_reason', 'Strategy-based')}")
                return True
            
            # Log specific error details
            if result is not None:
                logger.warning(f"Close attempt {attempt + 1}/{max_retries} failed: {result.comment} (retcode: {result.retcode})")
                
                # Don't retry on certain permanent errors
                permanent_errors = {
                    mt5.TRADE_RETCODE_INVALID_VOLUME,
                    mt5.TRADE_RETCODE_INVALID_PRICE,
                    mt5.TRADE_RETCODE_TRADE_DISABLED,
                    mt5.TRADE_RETCODE_MARKET_CLOSED
                }
                if result.retcode in permanent_errors:
                    logger.error(f"Permanent error, stopping retries: {result.comment}")
                    break
            else:
                logger.warning(f"Close attempt {attempt + 1}/{max_retries} failed: No result returned")
            
            # Brief pause before retry (except on last attempt)
            if attempt < max_retries - 1:
                time.sleep(1 + attempt * 0.5)  # Progressive backoff
        
        logger.error(f"Failed to close position {ticket} after {max_retries} attempts")
        return False

    except Exception as e:
        logger.error(f"Error closing position: {str(e)}")
        logger.error(f"Position data: {position}")
        logger.error(traceback.format_exc())
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




# def adjust_bot_emotions_based_on_performance():
#     """Example function to adjust emotions based on performance"""
#     try:
#         report = intelligent_engine.get_status_report()
        
#         for symbol, performance in report['symbol_performance'].items():
#             win_rate = performance.get('win_rate', 0)
#             daily_pnl = performance.get('daily_pnl', 0)
            
#             # Adjust emotion based on performance
#             if win_rate < 0.3 and daily_pnl < -50:
#                 # Poor performance - become more fearful
#                 new_emotion = EmotionLevel.FEARFUL
#                 logger.info(f"Adjusting {symbol} to FEARFUL due to poor performance")
                
#             elif win_rate < 0.4 and daily_pnl < -20:
#                 # Below average - become cautious
#                 new_emotion = EmotionLevel.CAUTIOUS
#                 logger.info(f"Adjusting {symbol} to CAUTIOUS due to below-average performance")
                
#             elif win_rate > 0.8 and daily_pnl > 100:
#                 # Excellent performance - become greedy
#                 new_emotion = EmotionLevel.GREEDY
#                 logger.info(f"Adjusting {symbol} to GREEDY due to excellent performance")
                
#             elif win_rate > 0.6 and daily_pnl > 50:
#                 # Good performance - become confident
#                 new_emotion = EmotionLevel.CONFIDENT
#                 logger.info(f"Adjusting {symbol} to CONFIDENT due to good performance")
                
#             else:
#                 # Normal performance - stay standard
#                 new_emotion = EmotionLevel.STANDARD
#                 continue  # No change needed
            
#             intelligent_engine.update_symbol_emotion(symbol, new_emotion)
        
#         return True
        
#     except Exception as e:
#         logger.error(f"Error adjusting emotions: {str(e)}")
#         return False

def adjust_bot_emotions_based_on_performance(self):
    """Adjust each symbol's emotion based on performance"""
    try:
        report = self.get_status_report()  # Should return symbol performance stats

        for symbol, performance in report['symbol_performance'].items():
            win_rate = performance.get('win_rate', 0)
            daily_pnl = performance.get('daily_pnl', 0)
            
            new_emotion = EmotionLevel.STANDARD  # default

            if win_rate < 0.3 and daily_pnl < -50:
                new_emotion = EmotionLevel.FEARFUL
                logger.info(f"Adjusting {symbol} to FEARFUL due to poor performance")
            elif win_rate < 0.4 and daily_pnl < -20:
                new_emotion = EmotionLevel.CAUTIOUS
                logger.info(f"Adjusting {symbol} to CAUTIOUS due to below-average performance")
            elif win_rate > 0.8 and daily_pnl > 100:
                new_emotion = EmotionLevel.GREEDY
                logger.info(f"Adjusting {symbol} to GREEDY due to excellent performance")
            elif win_rate > 0.6 and daily_pnl > 50:
                new_emotion = EmotionLevel.CONFIDENT
                logger.info(f"Adjusting {symbol} to CONFIDENT due to good performance")
            
            # Update emotion only if it changed
            if self.symbol_configs[symbol].emotion != new_emotion:
                self.update_symbol_emotion(symbol, new_emotion)
                
    except Exception as e:
        logger.error(f"Error adjusting emotions: {str(e)}")



def get_detailed_status():
    """Get detailed status of all symbols and strategies"""
    try:
        report = intelligent_engine.get_status_report()
        
        print("\n" + "="*80)
        print("INTELLIGENT MULTI-SYMBOL TRADING BOT STATUS")
        print("="*80)
        
        # Global stats
        global_stats = report['global_stats']
        print(f"Active Symbols: {global_stats['active_symbols']}")
        print(f"Total Open Positions: {global_stats['total_positions']}")
        print(f"Daily Profit: ${global_stats['daily_profit']:.2f}")
        print(f"Total Trades Today: {global_stats['total_trades']}")
        
        # Strategy distribution
        print(f"\nStrategy Distribution:")
        for strategy, count in report['strategy_distribution'].items():
            print(f"  {strategy.title()}: {count} symbols")
        
        # Emotion distribution  
        print(f"\nEmotion Distribution:")
        for emotion, count in report['emotion_distribution'].items():
            print(f"  {emotion.title()}: {count} symbols")
        
        # Individual symbol performance
        print(f"\nSymbol Performance:")
        print("-" * 80)
        for symbol, perf in report['symbol_performance'].items():
            status = "✅" if perf['enabled'] else "❌"
            positions_info = f"{perf['current_positions']}/{perf['max_positions']}"
            win_rate_str = f"{perf['win_rate']:.1%}" if perf['win_rate'] > 0 else "N/A"
            
            print(f"{status} {symbol:8} | {perf['strategy']:10} | {perf['emotion']:8} | "
                  f"Pos: {positions_info:5} | Trades: {perf['daily_trades']:2} | "
                  f"P&L: ${perf['daily_pnl']:7.2f} | WR: {win_rate_str:6}")
        
        # Active positions
        if report['active_positions']:
            print(f"\nActive Positions:")
            print("-" * 60)
            for symbol, positions in report['active_positions'].items():
                for pos in positions:
                    profit_str = f"${pos['profit']:.2f} ({pos['profit_pct']:.1f}%)"
                    print(f"{symbol:8} | {pos['type']:4} | {pos['volume']:5} lots | "
                          f"Ticket: {pos['ticket']} | Profit: {profit_str}")
        
        print("="*80 + "\n")
        
        return report
        
    except Exception as e:
        logger.error(f"Error getting detailed status: {str(e)}")
        return None


# Main trading loop - replaces the old smart_trading_loop
async def intelligent_multi_symbol_trading_loop():
    """Main entry point for the intelligent multi-symbol trading system"""
    try:
        # Setup default symbols if none configured
        if not intelligent_engine.symbol_configs:
            logger.info("No symbols configured, setting up default multi-symbol configuration...")
            intelligent_engine.setup_multi_symbol_trading()
        
        # Start the main trading loop
        await intelligent_engine.run_intelligent_trading_loop()
        
    except Exception as e:
        logger.error(f"Error in main trading loop: {str(e)}")
        state.running = False


# Replace the old trading_loop assignment
trading_loop = intelligent_multi_symbol_trading_loop