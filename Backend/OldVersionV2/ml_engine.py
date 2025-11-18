import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Dict
from config import logger
import joblib
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field
class ForexModel:
    """Machine learning model for forex trading predictions."""
    
    MODEL_PATH = Path("models/forex_model.joblib")
    MIN_SAMPLES = 100
    
    def __init__(self, load_existing: bool = False):
        """Initialize the forex prediction model."""
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        self.feature_columns: List[str] = []
        self.target_column: str = "target"
        
        if load_existing and self.MODEL_PATH.exists():
            self._load_model()

    def train(self, data: pd.DataFrame) -> Tuple[Optional[Dict[str, float]], List[Tuple[str, float]]]:
        """
        Train the model using time series cross-validation.
        
        Returns:
            Tuple containing metrics dictionary and feature importances
        """
        try:
            if len(data) < self.MIN_SAMPLES:
                logger.error(f"Insufficient data: {len(data)} samples")
                return None, []

            X, y = self._prepare_data(data)
            tscv = TimeSeriesSplit(n_splits=5)
            metrics = {'accuracy': [], 'precision': [], 'recall': []}
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                self.model.fit(X_train_scaled, y_train)
                y_pred = self.model.predict(X_test_scaled)
                
                metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
                metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            
            avg_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
            self._save_model()
            return avg_metrics, self._get_feature_importances()
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            return None, []

    def predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate prediction and confidence score for the latest data point.
        
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            if not self.feature_columns or not hasattr(self.model, 'estimators_'):
                raise ValueError("Model not trained")
                
            X = data[self.feature_columns]
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)[-1]
            confidence = float(np.max(self.model.predict_proba(X_scaled)[-1]))
            
            if not 0 <= confidence <= 1:
                raise ValueError(f"Invalid confidence score: {confidence}")
                
            return int(prediction), confidence
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return 0, 0.0

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for model training."""
        data = data.dropna()
        self.feature_columns = [col for col in data.columns if col != self.target_column]
        return data[self.feature_columns], data[self.target_column]

    def _get_feature_importances(self) -> List[Tuple[str, float]]:
        """Get sorted list of feature importances."""
        return sorted(
            zip(self.feature_columns, self.model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
        
    def _save_model(self) -> None:
        """Save trained model to disk."""
        self.MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler}, self.MODEL_PATH)
        
    def _load_model(self) -> None:
        """Load trained model from disk."""
        loaded = joblib.load(self.MODEL_PATH)
        self.model = loaded['model']
        self.scaler = loaded['scaler']

@dataclass
class FeatureEngineer:
    """Generate and engineer features for forex trading model."""
    
    def __init__(self):
        """Initialize feature engineer with default parameters."""
        self.min_periods = {
            'short': 10,
            'medium': 25,
            'long': 50
        }
        self.default_timezone = 'UTC'

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data into feature-rich dataset."""
        try:
            if not self._validate_input(df):
                logger.error("Invalid input data")
                return pd.DataFrame()

            df = df.copy()
            df = self._add_basic_features(df)
            df = self._add_technical_indicators(df)
            df = self._add_temporal_features(df)
            df = self._add_statistical_features(df)
            
            # Add target column - MOST IMPORTANT FIX
            df = self._create_target_column(df)
            
            df = self._clean_data(df)
            logger.info(f"Feature engineering completed: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}", exc_info=True)
            return pd.DataFrame()
        
    def _validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input DataFrame has required columns and data."""
        required = {'open', 'high', 'low', 'close'}
        return (isinstance(df, pd.DataFrame) and 
                not df.empty and 
                required.issubset(set(df.columns)))

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fundamental price-based features."""
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['close'].rolling(
            window=20, 
            min_periods=self.min_periods['short']
        ).std()
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['body_size'] = (df['close'] - df['open']) / df['close']
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators."""
        df['SMA_short'] = df['close'].rolling(
            window=20, 
            min_periods=self.min_periods['short']
        ).mean()
        df['SMA_long'] = df['close'].rolling(
            window=50, 
            min_periods=self.min_periods['medium']
        ).mean()
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['RSI'] = self._calculate_rsi(df)
        df['MACD'], df['MACD_signal'] = self._calculate_macd(df)
        df['ATR'] = self._calculate_atr(df)
        return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        try:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(
                window=period, 
                min_periods=period//2
            ).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(
                window=period, 
                min_periods=period//2
            ).mean()
            rs = gain / loss.replace(0, np.finfo(float).eps)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"RSI calculation failed: {str(e)}")
            return pd.Series()

    def _calculate_macd(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD line and signal line."""
        ema_short = df['close'].ewm(span=12, adjust=False).mean()
        ema_long = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=period//2).mean()

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df.index = pd.to_datetime(df.index, utc=True)
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_london_open'] = df['hour'].isin([7, 8, 9, 10, 11]).astype(int)
        df['is_ny_open'] = df['hour'].isin([13, 14, 15, 16, 17]).astype(int)
        df['is_tokyo_open'] = df['hour'].isin([0, 1, 2, 3, 4]).astype(int)
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical momentum features."""
        df['momentum'] = df['close'].diff(5)
        df['z_score'] = (df['close'] - df['close'].rolling(
            window=20, 
            min_periods=self.min_periods['short']
        ).mean()) / df['volatility']
        df['volume_trend'] = df['volume'].rolling(
            window=20, 
            min_periods=self.min_periods['short']
        ).mean() if 'volume' in df.columns else 0
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize dataset."""
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna(thresh=len(df.columns) * 0.8)  # Drop rows with >20% NaN
        return df.ffill().bfill()  # Forward then backward fill remaining NaNs
    
    def _create_target_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable for prediction (price movement 1 period ahead)."""
        # Create binary target: 1 if price increases, 0 otherwise
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        return df