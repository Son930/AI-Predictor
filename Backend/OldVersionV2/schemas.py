from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, List

class TradeRequest(BaseModel):
    symbol: str
    action: str  # "buy" or "sell"
    lot_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class TradeSignal(BaseModel):
    action: str
    confidence: float
    features: Dict[str, float]
    timestamp: datetime

class FeatureData(BaseModel):
    features: Dict[str, float]
    timestamp: datetime

class PositionInfo(BaseModel):
    ticket: int
    time: datetime
    type: int
    symbol: str
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float

class BotConfig(BaseModel):
    max_loss_percent: float = 1.2
    min_confidence: float = 0.65
    risk_per_trade: float = 0.02

class BotStateResponse(BaseModel):
    running: bool
    paused: bool
    symbol: str
    positions: List[PositionInfo]
    last_signal: int
    error_count: int
    last_error: str

class SymbolRequest(BaseModel):
    symbol: str

class PositionResponse(BaseModel):
    positions: List[dict]
    timestamp: float

class ClosePositionResponse(BaseModel):
    success: bool
    message: str