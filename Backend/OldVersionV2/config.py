# Configuration and State Management of BOT 
# V1.0.1

from asyncio.log import logger
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
import threading


@dataclass
class BotState:
    """Manages the state and configuration of the trading bot."""
    # Core state
    running: bool = False
    paused: bool = False
    model: Any = None
    
    # Trading metrics
    last_prediction: int = 0
    last_confidence: float = 0.0
    last_features: Dict[str, float] = field(default_factory=dict)
    positions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Trading configuration
    symbol: str = "XAUUSD"
    timeframe: str = "M1"
    num_bars: int = 300
    lot_size: float = 0.1
    suggested_lot_size: float = 0.1
    max_loss_percent: float = 1.2
    min_confidence: float = 0.65
    
    # System metrics
    error_count: int = 0
    last_error: Optional[str] = None
    volatility: float = 0.0
    last_data_refresh: datetime = field(default_factory=datetime.now)
    model_version: str = "1.0.1"
    
    def __post_init__(self):
        """Validate initial state."""
        self._validate_config()
        self._lock = threading.Lock()  # Thread safety
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.lot_size <= 100:
            raise ValueError(f"Invalid lot_size: {self.lot_size}")
        if not 0 < self.max_loss_percent <= 100:
            raise ValueError(f"Invalid max_loss_percent: {self.max_loss_percent}")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError(f"Invalid min_confidence: {self.min_confidence}")
        if self.num_bars <= 0:
            raise ValueError(f"Invalid num_bars: {self.num_bars}")

    def update_position(self, position: Dict[str, Any]) -> None:
        """Thread-safe position update."""
        with self._lock:
            self.positions.append(position)
            
    def clear_positions(self) -> None:
        """Thread-safe position clearing."""
        with self._lock:
            self.positions.clear()

    def update_metrics(self, prediction: int, confidence: float, 
                      features: Dict[str, float]) -> None:
        """Thread-safe update of trading metrics."""
        with self._lock:
            self.last_prediction = prediction
            self.last_confidence = confidence
            self.last_features = features.copy()
            self.last_data_refresh = datetime.now()

    def record_error(self, error: str) -> None:
        """Thread-safe error recording."""
        with self._lock:
            self.error_count += 1
            self.last_error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        data = asdict(self)
        data['last_data_refresh'] = data['last_data_refresh'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BotState':
        """Create state from dictionary."""
        data['last_data_refresh'] = datetime.fromisoformat(data['last_data_refresh'])
        return cls(**data)

    def save_state(self):
        state_data = {
            'running': self.running,
            'paused': self.paused,
            'positions': [p.__dict__ for p in self.positions],
            # Exclude non-serializable items
            # 'model': self.model,  # Remove this line
            'last_prediction': self.last_prediction.isoformat() if self.last_prediction else None,
            'error_count': self.error_count
        }
        
        try:
            with open('state.json', 'w') as f:
                json.dump(state_data, f)
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")

    @classmethod
    def load_state(cls, filepath: str = "bot_state.json") -> 'BotState':
        """Load state from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")
            return cls()

state = BotState()

def configure_logging(log_file: str = "smart_trading_bot.log", 
                     log_level: int = logging.INFO) -> None:
    """
    Configure logging with file and console output.
    
    Args:
        log_file: Path to log file
        log_level: Logging level (e.g., logging.INFO)
    """
    try:
        # Remove any existing handlers to prevent duplicate logging
        logger.handlers.clear()
        
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger.info("Logging configured successfully")
    except Exception as e:
        print(f"Failed to configure logging: {str(e)}")  # Fallback if logging fails

if __name__ == "__main__":
    # Example usage
    configure_logging()
    state.update_metrics(1, 0.85, {"price": 1800.5, "rsi": 65.2})
    state.record_error("Connection timeout")
    state.save_state()
    loaded_state = BotState.load_state()
    logger.info(f"Loaded state: {loaded_state.to_dict()}")