# symbol_manager.py
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import yaml
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SymbolProfile:
    volatility: float
    trend_strength: float
    correlation: Dict[str, float]  # Correlation with other symbols
    liquidity_score: float

class SymbolAnalyzer:
    """Analyzes and classifies symbols based on market behavior"""
    
    def __init__(self):
        self.symbol_profiles = {}
        
    def analyze_all_symbols(self, min_liquidity=1000000) -> List[str]:
        """Analyze all available symbols and return tradable ones"""
        all_symbols = mt5.symbols_get()
        if not all_symbols:
            logger.error("Failed to fetch symbols from MT5")
            return []

        tradable_symbols = []
        for s in all_symbols:
            try:
                if (s.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL and 
                    s.visible and 
                    s.volume_real > min_liquidity):
                    
                    profile = SymbolProfile(
                        volatility=self.calculate_volatility(s.name),
                        trend_strength=self.calculate_trend_strength(s.name),
                        correlation={},
                        liquidity_score=s.volume_real
                    )
                    self.symbol_profiles[s.name] = profile
                    tradable_symbols.append(s.name)
                    
            except Exception as e:
                logger.error(f"Error analyzing {s.name}: {str(e)}")
                
        return tradable_symbols

    def calculate_volatility(self, symbol: str, lookback=100) -> float:
        """Calculate normalized volatility for a symbol"""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, lookback)
        if rates is None:
            return 0
            
        closes = np.array([r[4] for r in rates])
        returns = np.diff(closes) / closes[:-1]
        return np.std(returns) * np.sqrt(24)  # Annualized

    def calculate_trend_strength(self, symbol: str) -> float:
        """Calculate ADX-based trend strength (0-100)"""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 14)
        if rates is None or len(rates) < 14:
            return 0
            
        # Simplified ADX calculation
        highs = np.array([r[2] for r in rates])
        lows = np.array([r[3] for r in rates])
        closes = np.array([r[4] for r in rates])
        
        plus_dm = highs[1:] - highs[:-1]
        minus_dm = lows[:-1] - lows[1:]
        
        tr = np.maximum(highs[1:] - lows[1:], 
                       np.maximum(abs(highs[1:] - closes[:-1]), 
                                   abs(lows[1:] - closes[:-1])))
        
        plus_di = 100 * np.mean(plus_dm[plus_dm > minus_dm] / tr[plus_dm > minus_dm])
        minus_di = 100 * np.mean(minus_dm[minus_dm > plus_dm] / tr[minus_dm > plus_dm])
        
        return abs(plus_di - minus_di)  # Simplified trend strength

class StrategyAllocator:
    """Dynamically allocates strategies based on symbol characteristics"""
    
    def __init__(self):
        self.strategy_rules = self.load_strategy_rules()
        
    def load_strategy_rules(self, filepath="symbol_strategies.yaml"):
        """Load strategy allocation rules from config"""
        try:
            with open(filepath) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Strategy rules not found, using defaults")
            return {
                'high_volatility': {
                    'strategy': 'SCALPING',
                    'emotion': 'CONFIDENT',
                    'max_positions': 3,
                    'lot_size': 0.02
                },
                'medium_volatility': {
                    'strategy': 'BALANCED', 
                    'emotion': 'STANDARD',
                    'max_positions': 2,
                    'lot_size': 0.01
                },
                'low_volatility': {
                    'strategy': 'SWING',
                    'emotion': 'CAUTIOUS',
                    'max_positions': 1,
                    'lot_size': 0.005
                }
            }
    
    def get_strategy_for_symbol(self, symbol: str, profile: SymbolProfile):
        """Determine optimal strategy based on symbol profile"""
        volatility = profile.volatility
        
        if volatility > 0.15:  # Highly volatile
            return self.strategy_rules['high_volatility']
        elif volatility > 0.08:  # Medium volatility
            return self.strategy_rules['medium_volatility']
        else:  # Low volatility
            return self.strategy_rules['low_volatility']