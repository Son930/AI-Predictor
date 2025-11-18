import logging
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
from config import state, configure_logging
from schemas import TradeRequest, PositionInfo
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from threading import Lock

logger = logging.getLogger(__name__)

class MT5ConnectionError(Exception):
    """Custom exception for MT5 connection issues."""
    pass

class MT5Manager:
    """Manage MetaTrader 5 terminal connection and trading operations."""
    
    TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "D1": mt5.TIMEFRAME_D1
    }
    
    def __init__(self, login: Optional[int] = None, password: Optional[str] = None, 
                 server: Optional[str] = None, timeout: int = 30000):
        """
        Initialize MT5 Manager with optional login credentials.
        
        Args:
            login: MT5 account number
            password: MT5 account password
            server: MT5 server name
            timeout: Connection timeout in milliseconds
        """
        self.connected = False
        self.login = login
        self.password = password
        self.server = server
        self.timeout = timeout
        self._lock = Lock()  # Thread safety for MT5 operations
        self._last_connection_check = None
        self._connection_check_interval = timedelta(minutes=5)

    @retry(stop=stop_after_attempt(5), 
           wait=wait_exponential(multiplier=1, min=4, max=60),
           retry=retry_if_exception_type(Exception))
    def initialize(self) -> bool:
        """Initialize and verify MT5 connection with retry logic."""
        with self._lock:
            try:
                init_params = {"timeout": self.timeout}
                if self.login and self.password and self.server:
                    init_params.update({
                        "login": self.login,
                        "password": self.password,
                        "server": self.server
                    })
                
                if not mt5.initialize(**init_params):
                    raise MT5ConnectionError("Initialization failed")
                    
                if not self._verify_connection():
                    mt5.shutdown()
                    raise MT5ConnectionError("Connection verification failed")
                    
                self.connected = True
                self._last_connection_check = datetime.now()
                logger.info(f"MT5 connection established: Account #{mt5.account_info().login}")
                return True
                
            except Exception as e:
                logger.error(f"Connection attempt failed: {str(e)}")
                raise
                
    def _verify_connection(self) -> bool:
        """Verify MT5 connection is active."""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return False
            return True
        except Exception:
            return False

    def _ensure_connection(self) -> None:
        """Ensure connection is active, reconnect if needed."""
        current_time = datetime.now()
        if (not self.connected or 
            (self._last_connection_check and 
             current_time - self._last_connection_check > self._connection_check_interval)):
            logger.info("Verifying connection status")
            if not self._verify_connection():
                logger.warning("Connection lost, attempting to reconnect")
                self.initialize()
            self._last_connection_check = current_time

    def fetch_historical_data(self, symbol: str, timeframe: str, 
                            num_bars: int) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data from MT5.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe string (e.g., 'M1', 'H1')
            num_bars: Number of bars to fetch
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            self._ensure_connection()
            if timeframe not in self.TIMEFRAME_MAP:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
                
            if not mt5.symbol_info(symbol):
                raise ValueError(f"Invalid symbol: {symbol}")
                
            rates = mt5.copy_rates_from_pos(
                symbol,
                self.TIMEFRAME_MAP[timeframe],
                0,
                num_bars
            )
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            return df.set_index('time')[['open', 'high', 'low', 'close', 'tick_volume']]
            
        except Exception as e:
            logger.error(f"Data fetch error for {symbol}: {str(e)}")
            return None

    def execute_order(self, request: TradeRequest) -> Dict[str, Union[bool, str]]:
        """
        Execute a market order on MT5.
        
        Returns:
            Dict with success status and message
        """
        try:
            self._ensure_connection()
            symbol_info = mt5.symbol_info(request.symbol)
            if not symbol_info or not symbol_info.visible:
                return {"success": False, "message": f"Invalid/unavailable symbol: {request.symbol}"}

            tick = mt5.symbol_info_tick(request.symbol)
            if not tick:
                return {"success": False, "message": "Failed to get current price"}

            order_type = mt5.ORDER_TYPE_BUY if request.action == "buy" else mt5.ORDER_TYPE_SELL
            price = tick.ask if request.action == "buy" else tick.bid
            
            order = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": request.symbol,
                "volume": float(request.lot_size),
                "type": order_type,
                "price": price,
                "sl": request.stop_loss or 0.0,
                "tp": request.take_profit or 0.0,
                "deviation": 20,
                "magic": int(time.time() * 1000),  # Unique order ID
                "comment": "AI Trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC
            }

            with self._lock:
                result = mt5.order_send(order)
                
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                message = getattr(result, 'comment', 'Unknown error')
                logger.error(f"Order failed: {message}")
                return {"success": False, "message": message}
                
            logger.info(f"Order executed: {result.order}")
            return {"success": True, "message": f"Order #{result.order} executed"}
            
        except Exception as e:
            logger.error(f"Order execution failed: {str(e)}")
            return {"success": False, "message": str(e)}

    def get_positions(self, symbol: Optional[str] = None) -> List[PositionInfo]:
        """Get current open positions."""
        try:
            self._ensure_connection()
            positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
            if not positions:
                return []
                
            return [PositionInfo(
                ticket=p.ticket,
                time=datetime.fromtimestamp(p.time),
                type=p.type,
                symbol=p.symbol,
                volume=p.volume,
                price_open=p.price_open,
                price_current=p.price_current,
                sl=p.sl,
                tp=p.tp,
                profit=p.profit
            ) for p in positions]
            
        except Exception as e:
            logger.error(f"Failed to get positions: {str(e)}")
            return []

    def close(self) -> None:
        """Close MT5 connection."""
        with self._lock:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                logger.info("MT5 connection closed")