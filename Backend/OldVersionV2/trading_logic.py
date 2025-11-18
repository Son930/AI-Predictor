from typing import Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from mt5_core import MT5Manager
from ml_engine import ForexModel, FeatureEngineer
from config import state, logger
from schemas import TradeSignal

class TradingStrategy:
    """Manages trading signals and position sizing for forex trading."""
    
    MIN_DATA_POINTS = 100
    DEFAULT_LOT_SIZE = 0.1
    MAX_LOT_SIZE = 5.0
    RISK_PERCENT = 0.02

    def __init__(self):
        """Initialize trading strategy components."""
        self.mt5 = MT5Manager()
        self.model = ForexModel()
        self.feature_engineer = FeatureEngineer()
        self.last_trade_time: Optional[datetime] = None
        self.min_trade_interval = timedelta(minutes=15)

    def generate_signal(self) -> TradeSignal:
        """
        Generate a trading signal based on market data and model prediction.
        
        Returns:
            TradeSignal: Trading decision with confidence and metadata
        """
        try:
            df = self._get_processed_data()
            if df is None or len(df) < self.MIN_DATA_POINTS:
                logger.warning(f"Insufficient data points: {len(df) if df is not None else 0}")
                return TradeSignal(action="hold", confidence=0.0)

            prediction, confidence = self._predict_with_validation(df)
            action = self._interpret_prediction(prediction, confidence)
            
            return TradeSignal(
                action=action,
                confidence=float(confidence),
                features=df.iloc[-1].to_dict(),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Signal generation failed: {str(e)}", exc_info=True)
            return TradeSignal(action="hold", confidence=0.0)

    def _get_processed_data(self) -> Optional[pd.DataFrame]:
        """Fetch and process market data for analysis."""
        try:
            raw_data = self.mt5.fetch_historical_data(
                symbol=state.symbol,
                timeframe=state.timeframe,
                num_bars=state.num_bars
            )
            if raw_data is None or len(raw_data) == 0:
                logger.warning("No historical data received from MT5")
                return None

            processed_data = self.feature_engineer.process(raw_data)
            if processed_data.empty:
                return None

            processed_data['target'] = np.where(
                processed_data['close'].shift(-1) > processed_data['close'],
                1, 
                -1
            )
            return processed_data.dropna()
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            return None

    def _predict_with_validation(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Generate and validate model prediction."""
        prediction, confidence = self.model.predict(df)
        
        if not isinstance(prediction, (int, np.integer)) or prediction not in [-1, 1]:
            logger.warning(f"Invalid prediction value: {prediction}")
            return 0, 0.0
            
        if not 0 <= confidence <= 1:
            logger.warning(f"Invalid confidence value: {confidence}")
            return 0, 0.0
            
        return prediction, confidence

    def _interpret_prediction(self, prediction: int, confidence: float) -> str:
        """Convert model prediction to trading action."""
        if confidence < state.min_confidence:
            return "hold"
        
        # Add time-based restriction to prevent over-trading
        current_time = datetime.now()
        if (self.last_trade_time and 
            current_time - self.last_trade_time < self.min_trade_interval):
            return "hold"
            
        action = "buy" if prediction == 1 else "sell"
        if action != "hold":
            self.last_trade_time = current_time
        return action

    def calculate_position_size(self, account_balance: float, volatility: float) -> float:
        """
        Calculate position size based on risk management parameters.
        
        Args:
            account_balance: Current account balance in base currency
            volatility: Current market volatility measure
            
        Returns:
            float: Calculated lot size
        """
        try:
            if account_balance <= 0 or volatility <= 0:
                logger.warning("Invalid input for position sizing")
                return self.DEFAULT_LOT_SIZE

            risk_amount = account_balance * self.RISK_PERCENT
            price = self.mt5.get_current_price(state.symbol)
            pip_value = self._get_pip_value(state.symbol)
            
            if price <= 0 or pip_value <= 0:
                logger.warning(f"Invalid market data - price: {price}, pip_value: {pip_value}")
                return self.DEFAULT_LOT_SIZE
                
            position_size = risk_amount / (volatility * pip_value * price)
            return round(max(self.DEFAULT_LOT_SIZE, min(position_size, self.MAX_LOT_SIZE)), 2)
        except Exception as e:
            logger.error(f"Position size calculation failed: {str(e)}")
            return self.DEFAULT_LOT_SIZE

    def _get_pip_value(self, symbol: str) -> float:
        """Determine pip value based on symbol characteristics."""
        symbol = symbol.upper()
        if "JPY" in symbol:
            return 0.01
        if any(x in symbol for x in ["XAU", "XAG"]):
            return 0.1
        return 0.0001