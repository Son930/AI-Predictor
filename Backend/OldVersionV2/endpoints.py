# endpoints.py
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional, Union

from pydantic import conset
from config import state, BotState, logger
from schemas import (TradeRequest, BotConfig, BotStateResponse, PositionInfo, 
                    TradeSignal, FeatureData, SymbolRequest, PositionResponse, 
                    ClosePositionResponse)
from trading_logic import TradingStrategy
from mt5_core import MT5Manager
from ml_engine import ForexModel, FeatureEngineer
import pandas as pd
from datetime import datetime
import time

router = APIRouter()

# Dependencies
def get_mt5_manager() -> MT5Manager:
    from .main import mt5_manager
    if mt5_manager is None:
        raise HTTPException(status_code=503, detail="MT5 Manager not initialized")
    return mt5_manager

def get_forex_model() -> ForexModel:
    from .main import trading_strategy
    if trading_strategy is None or trading_strategy.model is None:
        raise HTTPException(status_code=503, detail="Forex Model not initialized")
    return trading_strategy.model

def get_feature_engineer() -> FeatureEngineer:
    from .main import trading_strategy
    if trading_strategy is None or trading_strategy.feature_engineer is None:
        raise HTTPException(status_code=503, detail="Feature Engineer not initialized")
    return trading_strategy.feature_engineer


@router.post("/start")
async def start_bot(mt5: MT5Manager = Depends(get_mt5_manager)):
    try:
        if state.running:
            raise HTTPException(status_code=400, detail="Bot is already running")
        if not mt5.connected and not mt5.initialize():
            logger.error("MT5 initialization failed during start")
            raise HTTPException(status_code=500, detail="Failed to connect to MT5")
        state.running = True
        state.paused = False
        logger.info("Trading bot started")
        return {"message": "Trading bot started", "running": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/stop")
async def stop_bot(mt5: MT5Manager = Depends(get_mt5_manager)):
    if not state.running:
        raise HTTPException(status_code=400, detail="Bot is not running")
    state.running = False
    state.paused = False
    mt5.close()
    logger.info("Trading bot stopped")
    return {"message": "Trading bot stopped", "running": False}

@router.post("/pause")
async def pause_bot():
    if not state.running:
        raise HTTPException(status_code=400, detail="Bot is not running")
    if state.paused:
        raise HTTPException(status_code=400, detail="Bot is already paused")
    state.paused = True
    logger.info("Trading bot paused")
    return {"message": "Trading bot paused", "paused": True}

@router.post("/resume")
async def resume_bot():
    if not state.running:
        raise HTTPException(status_code=400, detail="Bot is not running")
    if not state.paused:
        raise HTTPException(status_code=400, detail="Bot is not paused")
    state.paused = False
    logger.info("Trading bot resumed")
    return {"message": "Trading bot resumed", "paused": False}

# State and Configuration Endpoints
@router.get("/state", response_model=BotStateResponse)
async def get_state(mt5: MT5Manager = Depends(get_mt5_manager)):
    try:
        positions = mt5.get_positions(state.symbol)
        return BotStateResponse(
            running=state.running,
            paused=state.paused,
            symbol=state.symbol,
            positions=[PositionInfo(**pos.__dict__) for pos in positions],
            last_signal=state.last_prediction,
            error_count=state.error_count,
            last_error=state.last_error or "None"
        )
    except Exception as e:
        logger.error(f"Failed to get state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get state: {str(e)}")

@router.post("/configure")
async def update_config(config: BotConfig):
    try:
        if config.max_loss_percent <= 0 or config.max_loss_percent > 100:
            raise ValueError("max_loss_percent must be between 0 and 100")
        if config.min_confidence < 0 or config.min_confidence > 1:
            raise ValueError("min_confidence must be between 0 and 1")
        state.max_loss_percent = config.max_loss_percent
        state.min_confidence = config.min_confidence
        state.save_state()
        logger.info(f"Configuration updated: max_loss={state.max_loss_percent}, "
                   f"min_confidence={state.min_confidence}")
        return {"message": "Configuration updated", "new_config": state.to_dict()}
    except Exception as e:
        logger.error(f"Configuration update failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")


# ForexModel Endpoints
@router.post("/train-model")
async def train_model(
    mt5: MT5Manager = Depends(get_mt5_manager),
    model: ForexModel = Depends(get_forex_model),
    feature_engineer: FeatureEngineer = Depends(get_feature_engineer)
):
    try:
        df = mt5.fetch_historical_data(state.symbol, state.timeframe, state.num_bars)
        if df is None or df.empty:
            raise ValueError("No data available for training")
        processed_df = feature_engineer.process(df)
        metrics, importances = model.train(processed_df)
        if metrics is None:
            raise ValueError("Training failed")
        state.model = model
        state.last_data_refresh = datetime.now()
        logger.info(f"Model trained: {metrics}")
        return {
            "message": "Model trained successfully",
            "metrics": metrics,
            "feature_importances": importances
        }
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@router.get("/predict", response_model=Dict[str, Union[int, float]])
async def predict(
    mt5: MT5Manager = Depends(get_mt5_manager),
    model: ForexModel = Depends(get_forex_model),
    feature_engineer: FeatureEngineer = Depends(get_feature_engineer)
):
    try:
        df = mt5.fetch_historical_data(state.symbol, state.timeframe, state.num_bars)
        if df is None or df.empty:
            raise ValueError("No data available for prediction")
        processed_df = feature_engineer.process(df)
        prediction, confidence = model.predict(processed_df)
        return {"prediction": prediction, "confidence": confidence}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# FeatureEngineer Endpoints
@router.post("/process-features", response_model=FeatureData)
async def process_features(
    mt5: MT5Manager = Depends(get_mt5_manager),
    feature_engineer: FeatureEngineer = Depends(get_feature_engineer)
):
    try:
        df = mt5.fetch_historical_data(state.symbol, state.timeframe, state.num_bars)
        if df is None or df.empty:
            raise ValueError("No data available for feature processing")
        processed_df = feature_engineer.process(df)
        return FeatureData(
            features=processed_df.iloc[-1].to_dict(),
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Feature processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature processing failed: {str(e)}")

# MT5Manager Endpoints
@router.get("/historical-data")
async def get_historical_data(
    symbol: str = state.symbol,
    timeframe: str = state.timeframe,
    num_bars: int = state.num_bars,
    mt5: MT5Manager = Depends(get_mt5_manager)
):
    try:
        df = mt5.fetch_historical_data(symbol, timeframe, num_bars)
        if df is None or df.empty:
            raise ValueError("No historical data available")
        return {
            "message": "Data fetched successfully",
            "data": df.to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Data fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {str(e)}")

@router.get("/positions")
async def get_positions(
    symbol: Optional[str] = None,
    mt5: MT5Manager = Depends(get_mt5_manager)
):
    try:
        positions = mt5.get_positions(symbol)
        return {
            "message": "Positions retrieved successfully",
            "positions": [pos.__dict__ for pos in positions]
        }
    except Exception as e:
        logger.error(f"Position retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Position retrieval failed: {str(e)}")

# Additional Endpoints from Your Request
@router.get("/data")
async def get_data(
    mt5: MT5Manager = Depends(get_mt5_manager),
    feature_engineer: FeatureEngineer = Depends(get_feature_engineer)
):
    """Get latest processed market data."""
    logger.info("Data requested")
    try:
        df = mt5.fetch_historical_data(state.symbol, state.timeframe, state.num_bars)
        if df is None or df.empty:
            return {"error": "Failed to fetch data"}
        # Assuming add_volume_column is handled by MT5Manager's fetch_historical_data
        # which already includes tick_volume
        processed_df = feature_engineer.process(df)
        return processed_df.tail(1).to_dict(orient="records")[0]
    except Exception as e:
        logger.error(f"Failed to get data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get data: {str(e)}")

@router.get("/symbols", response_model=List[str])
async def get_symbols(mt5: MT5Manager = Depends(get_mt5_manager)):
    """Get available trading symbols."""
    logger.info("Symbols requested")
    try:
        if not mt5.connected and not mt5.initialize():
            logger.error("MT5 not connected")
            return []
        symbols = mt5.symbols_get()
        return [s.name for s in symbols if s.visible]
    except Exception as e:
        logger.error(f"Error getting symbols: {str(e)}")
        return []

@router.post("/set_symbol")
async def set_symbol(
    request: SymbolRequest,
    mt5: MT5Manager = Depends(get_mt5_manager)
):
    """Set the trading symbol."""
    symbol = request.symbol
    logger.info(f"Setting symbol to {symbol}")
    try:
        if not mt5.connected and not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not connected")
        symbols = [s.name for s in mt5.symbols_get() if s.visible]
        if not symbols or symbol not in symbols:
            logger.error(f"Symbol {symbol} not available")
            raise HTTPException(status_code=400, detail=f"Symbol {symbol} not available")
        state.symbol = symbol
        state.save_state()
        return {"message": f"Symbol set to {symbol}"}
    except Exception as e:
        logger.error(f"Failed to set symbol: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to set symbol: {str(e)}")

@router.post("/set_lot_size")
async def set_lot_size(lot_size: float):
    """Set the trading lot size."""
    logger.info(f"Setting lot size to {lot_size}")
    try:
        if lot_size <= 0 or lot_size > 10:
            logger.error("Invalid lot size")
            raise HTTPException(status_code=400, detail="Invalid lot size. Must be between 0 and 10")
        state.lot_size = lot_size
        state.save_state()
        return {"message": f"Lot size set to {lot_size}"}
    except Exception as e:
        logger.error(f"Failed to set lot size: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to set lot size: {str(e)}")

@router.post("/use_suggested_lot_size")
async def use_suggested_lot_size():
    """Use the suggested lot size from TradingStrategy."""
    logger.info(f"Using suggested lot size: {state.suggested_lot_size}")
    try:
        if state.suggested_lot_size <= 0:
            raise HTTPException(status_code=400, detail="Suggested lot size not calculated yet")
        state.lot_size = state.suggested_lot_size
        state.save_state()
        return {"message": f"Lot size set to suggested value: {state.lot_size}"}
    except Exception as e:
        logger.error(f"Failed to use suggested lot size: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to use suggested lot size: {str(e)}")

@router.get("/logs")
async def get_logs():
    """Get recent log entries."""
    logger.info("Logs requested")
    try:
        with open("smart_trading_bot.log", "r") as log_file:
            logs = log_file.readlines()
        return {"logs": logs[-150:]}  # Last 150 lines
    except Exception as e:
        logger.error(f"Failed to read logs: {str(e)}")
        return {"error": f"Failed to fetch logs: {str(e)}"}

@router.post("/manual_trade")
async def manual_trade(
    action: str,
    lot_size: Optional[float] = None,
    mt5: MT5Manager = Depends(get_mt5_manager)
):
    """Execute a manual trade."""
    logger.info(f"Manual trade requested: {action}")
    try:
        if action not in ["buy", "sell"]:
            raise HTTPException(status_code=400, detail="Invalid action. Must be 'buy' or 'sell'")
        
        lot_size = lot_size if lot_size is not None else state.lot_size
        if lot_size <= 0 or lot_size > 10:
            raise HTTPException(status_code=400, detail="Invalid lot size")
            
        request = TradeRequest(symbol=state.symbol, action=action, lot_size=lot_size)
        result = mt5.execute_order(request)
        if not result["success"]:
            logger.error(f"Manual trade failed: {result['message']}")
            raise HTTPException(status_code=500, detail=result["message"])
            
        return {"message": f"Manual {action} trade executed successfully"}
    except Exception as e:
        state.record_error(str(e))
        raise HTTPException(status_code=500, detail=f"Manual trade failed: {str(e)}")

@router.get("/open_positions", response_model=PositionResponse)
async def return_open_positions(mt5: MT5Manager = Depends(get_mt5_manager)):
    """Get open positions for the current symbol."""
    try:
        positions = mt5.get_positions(state.symbol)
        return {
            "positions": [pos.__dict__ for pos in positions],
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get open positions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get open positions: {str(e)}")

@router.post("/close_position/{ticket}", response_model=ClosePositionResponse)
async def api_close_position(
    ticket: int,
    mt5: MT5Manager = Depends(get_mt5_manager)
):
    """Close a specific position by ticket number."""
    try:
        positions = mt5.get_positions(state.symbol)
        for pos in positions:
            if pos.ticket == ticket:
                # Construct close order (opposite of original)
                action = "sell" if pos.type == mt5.ORDER_TYPE_BUY else "buy"
                request = TradeRequest(
                    symbol=pos.symbol,
                    action=action,
                    lot_size=pos.volume
                )
                result = mt5.execute_order(request)
                if result["success"]:
                    logger.info(f"Closed position {ticket}")
                    return {"success": True, "message": f"Closed position {ticket}"}
                else:
                    logger.error(f"Failed to close position {ticket}: {result['message']}")
                    return {"success": False, "message": f"Failed to close: {result['message']}"}
        logger.error(f"Position {ticket} not found")
        return {"success": False, "message": "Position not found"}
    except Exception as e:
        logger.error(f"Failed to close position {ticket}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to close position: {str(e)}")