import time
from functools import wraps
from typing import Callable, Any
from .config import logger

def timeit(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} executed in {duration:.2f}s")
        return result
    return wrapper

def handle_errors(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def validate_symbol(symbol: str) -> bool:
    valid_symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
    return symbol in valid_symbols

def calculate_volatility(df, window=20) -> float:
    returns = df['close'].pct_change()
    return returns.rolling(window).std().iloc[-1]