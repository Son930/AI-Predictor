from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime, time as dt_time
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pandas as pd
import numpy as np
import time
import MetaTrader5 as mt5
from bot_state import state, logger
from mt5_utils import fetch_historical_data, get_open_positions, check_mt5_connection, initialize_mt5
from trading_logic import trading_loop, evaluate_positions, close_position, execute_trade
from trading_logic import TradingStrategy, EmotionLevel, intelligent_engine
import feature_engineering as fe
import sqlite3

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# -----------------
# Market-hours scheduler
# -----------------
scheduler = BackgroundScheduler()

def _auto_start_bot():
    try:
        if not state.running:
            logger.info("Scheduler: starting bot for market hours")
            state.running = True
            state.paused = False
            state.error_count = 0
            state.last_error = None
            # Start trading loop on event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.create_task(trading_loop())
    except Exception as e:
        logger.error(f"Scheduler start error: {str(e)}")

def _auto_stop_bot():
    try:
        if state.running:
            logger.info("Scheduler: stopping bot after market hours")
            state.running = False
    except Exception as e:
        logger.error(f"Scheduler stop error: {str(e)}")

def _init_scheduler():
    try:
        if scheduler.running:
            return
        # Example: Mon-Fri 07:00 start, 22:00 stop server local time
        scheduler.add_job(_auto_start_bot, CronTrigger(day_of_week='mon-fri', hour=7, minute=0))
        scheduler.add_job(_auto_stop_bot, CronTrigger(day_of_week='mon-fri', hour=22, minute=0))
        scheduler.start()
        logger.info("APScheduler initialized: market hours Mon-Fri 07:00-22:00")
    except Exception as e:
        logger.error(f"Failed to init scheduler: {str(e)}")

class Mt5Credentials(BaseModel):
    login: int
    password: str
    server: str

class SymbolRequest(BaseModel):
    symbol: str

class PositionResponse(BaseModel):
    positions: List[dict]
    timestamp: float

class ClosePositionResponse(BaseModel):
    success: bool
    message: str

class TradeHistory(BaseModel):
    ticket: int
    symbol: str
    action: str
    volume: float
    entry_price: float
    exit_price: float
    profit: float
    open_time: str
    close_time: str

class SymbolConfigRequest(BaseModel):
    symbol: str
    strategy: str  # Will be converted to TradingStrategy enum
    emotion: str   # Will be converted to EmotionLevel enum
    max_positions: int
    custom_lot_size: Optional[float] = None
    custom_timeframe: Optional[int] = None

class UpdateEmotionRequest(BaseModel):
    symbol: str
    emotion: str

class UpdateStrategyRequest(BaseModel):
    symbol: str
    strategy: str

class ToggleSymbolRequest(BaseModel):
    symbol: str
    enabled: bool

class PerformanceReport(BaseModel):
    global_stats: Dict
    symbol_performance: Dict
    active_positions: Dict
    strategy_distribution: Dict
    emotion_distribution: Dict

def init_db():
    conn = sqlite3.connect("trade_history.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades
                 (ticket INTEGER, symbol TEXT, action TEXT, volume REAL, entry_price REAL,
                  exit_price REAL, profit REAL, open_time TEXT, close_time TEXT)''')
    conn.commit()
    conn.close()

init_db()
_init_scheduler()

@app.post("/connect_mt5")
async def connect_mt5(creds: Mt5Credentials):
    success = initialize_mt5(creds.login, creds.password, creds.server)
    if success:
        state.mt5_connected = True
        state.mt5_login = creds.login
        state.mt5_password = creds.password
        state.mt5_server = creds.server
        state.last_error = None
        logger.info("MT5 connection established successfully")
        return {"message": "MT5 connected successfully", "connected": True}
    else:
        state.mt5_connected = False
        state.last_error = "Failed to connect to MT5 with provided credentials"
        logger.error("MT5 connection failed")
        raise HTTPException(status_code=400, detail="Failed to connect to MT5")

@app.post("/start")
async def start_bot(background_tasks: BackgroundTasks):
    if state.running:
        return {"message": "Bot is already running"}
    logger.info("Bot start requested")
    state.running = True
    state.paused = False
    state.error_count = 0
    state.last_error = None
    background_tasks.add_task(trading_loop)
    return {"message": "Bot started"}

@app.post("/stop")
async def stop_bot():
    if not state.running:
        return {"message": "Bot is already stopped"}
    logger.info("Bot stop requested")
    state.running = False
    return {"message": "Bot stopped"}

@app.post("/pause")
async def pause_bot():
    if not state.running:
        return {"message": "Bot is not running"}
    state.paused = not state.paused
    logger.info(f"Bot {'paused' if state.paused else 'resumed'}")
    return {"message": f"Bot {'paused' if state.paused else 'resumed'}"}

@app.get("/status")
async def get_status():
    try:
        status = {
            "running": state.running,
            "paused": state.paused,
            "mt5_connected": state.mt5_connected,
            "symbol": state.symbol,
            "timeframe": int(state.timeframe),
            "num_bars": int(state.num_bars),
            "lot_size": float(state.lot_size),
            "last_prediction": int(state.last_prediction) if state.last_prediction is not None else None,
            "last_confidence": float(state.last_confidence) if state.last_confidence is not None else None,
            "suggested_lot_size": float(state.suggested_lot_size) if state.suggested_lot_size is not None else None,
            "error_count": int(state.error_count),
            "last_error": state.last_error
        }
        logger.info("Status requested")
        return status
    except Exception as e:
        logger.error(f"Error in get_status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mt5_status")
async def get_mt5_status():
    connected = check_mt5_connection()
    return {
        "connected": connected,
        "account_info": mt5.account_info()._asdict() if connected else None,
        "last_error": state.last_error
    }

@app.get("/data")
async def get_data():
    logger.info("Data requested")
    df = fetch_historical_data(state.symbol, state.timeframe, state.num_bars)
    if df is None or df.empty:
        logger.error("Failed to fetch data")
        return {"error": "Failed to fetch data"}
    df = fe.engineer_features(df)
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.error("engineer_features returned invalid data")
        return {"error": "Failed to process data"}
    try:
        result = df.tail(1).to_dict('records')[0]
        for key, value in result.items():
            if isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
                result[key] = float(value) if isinstance(value, (np.float64, np.float32)) else int(value)
        return result
    except Exception as e:
        logger.error(f"Error converting data to dict: {str(e)}")
        return {"error": "Failed to process data"}

@app.get("/symbols", response_model=List[str])
async def get_symbols():
    logger.info("Symbols requested")
    if not check_mt5_connection():
        return []
    try:
        symbols = mt5.symbols_get()
        return [s.name for s in symbols if s.visible]
    except Exception as e:
        logger.error(f"Error getting symbols: {str(e)}")
        return []

@app.post("/set_symbol")
async def set_symbol(request: SymbolRequest):
    symbol = request.symbol
    logger.info(f"Setting symbol to {symbol}")
    symbols = await get_symbols()
    if not symbols or symbol not in symbols:
        logger.error(f"Symbol {symbol} not available")
        return {"error": f"Symbol {symbol} not available"}
    state.symbol = symbol
    return {"message": f"Symbol set to {symbol}"}

@app.post("/set_lot_size")
async def set_lot_size(lot_size: float):
    logger.info(f"Setting lot size to {lot_size}")
    if lot_size <= 0 or lot_size > 10:
        logger.error("Invalid lot size")
        return {"error": "Invalid lot size"}
    state.lot_size = lot_size
    return {"message": f"Lot size set to {lot_size}"}

@app.post("/use_suggested_lot_size")
async def use_suggested_lot_size():
    logger.info(f"Using suggested lot size: {state.suggested_lot_size}")
    state.lot_size = state.suggested_lot_size
    return {"message": f"Lot size set to suggested value: {state.lot_size}"}

@app.get("/logs")
async def get_logs():
    logger.info("Logs requested")
    try:
        with open("smart_trading_bot.log", "r") as log_file:
            logs = log_file.readlines()
        return {"logs": logs[-150:]}
    except Exception as e:
        logger.error(f"Failed to read logs: {str(e)}")
        return {"error": "Failed to fetch logs"}

@app.post("/manual_trade")
async def manual_trade(action: str, lot_size: Optional[float] = None):
    logger.info(f"Manual trade requested: {action}")
    if action not in ["buy", "sell"]:
        return {"error": "Invalid action. Must be 'buy' or 'sell'"}
    if lot_size is None:
        lot_size = state.lot_size
    success = execute_trade(state.symbol, action, lot_size)
    if success:
        return {"message": f"Manual {action} trade executed successfully"}
    else:
        return {"error": f"Failed to execute manual {action} trade"}

@app.get("/open_positions", response_model=PositionResponse)
async def return_open_positions(symbol: Optional[str] = None):
    try:
        positions = evaluate_positions(symbol) if symbol else evaluate_positions()
        for pos in positions:
            if isinstance(pos.get('open_time'), pd.Timestamp):
                pos['open_time'] = pos['open_time'].isoformat()
            for key, value in list(pos.items()):
                if isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
                    pos[key] = float(value) if isinstance(value, (np.float64, np.float32)) else int(value)
        logger.info(f"Returning {len(positions)} positions")
        return {
            "positions": positions,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error in open_positions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/close_position/{ticket}", response_model=ClosePositionResponse)
async def api_close_position(ticket: int):
    try:
        positions = evaluate_positions(state.symbol)
        for pos in positions:
            if pos["ticket"] == ticket:
                success = close_position(pos)
                if success:
                    conn = sqlite3.connect("trade_history.db")
                    c = conn.cursor()
                    c.execute("INSERT INTO trades (ticket, symbol, action, volume, entry_price, exit_price, profit, open_time, close_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                              (pos['ticket'], pos['symbol'], pos['type'], pos['volume'], pos['entry_price'],
                               pos['current_price'], pos['profit'], pos['open_time'], pd.Timestamp.now().isoformat()))
                    conn.commit()
                    conn.close()
                return {"success": success, "message": f"Closed position {ticket}"}
        return {"success": False, "message": "Position not found"}
    except Exception as e:
        logger.error(f"Error in close_position: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trade_history", response_model=List[TradeHistory])
async def get_trade_history():
    try:
        conn = sqlite3.connect("trade_history.db")
        c = conn.cursor()
        c.execute("SELECT * FROM trades ORDER BY close_time DESC LIMIT 100")
        trades = c.fetchall()
        conn.close()
        return [TradeHistory(
            ticket=t[0], symbol=t[1], action=t[2], volume=t[3], entry_price=t[4],
            exit_price=t[5], profit=t[6], open_time=t[7], close_time=t[8]
        ) for t in trades]
    except Exception as e:
        logger.error(f"Error in get_trade_history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/add_symbol_config")
async def add_symbol_config(request: SymbolConfigRequest):
    """Add or update a symbol configuration with strategy and emotion"""
    try:
        # Convert string to enum
        try:
            strategy = TradingStrategy(request.strategy.lower())
            emotion = EmotionLevel(request.emotion.lower())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid strategy or emotion: {str(e)}")

        success = intelligent_engine.add_symbol(
            symbol=request.symbol,
            strategy=strategy,
            emotion=emotion,
            max_positions=request.max_positions,
            custom_lot_size=request.custom_lot_size,
            custom_timeframe=request.custom_timeframe
        )
        
        if success:
            return {"message": f"Symbol {request.symbol} configuration updated successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to configure symbol {request.symbol}")
            
    except Exception as e:
        logger.error(f"Error in add_symbol_config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_emotion")
async def update_emotion(request: UpdateEmotionRequest):
    """Update emotion level for a symbol"""
    try:
        try:
            emotion = EmotionLevel(request.emotion.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid emotion level")
            
        intelligent_engine.update_symbol_emotion(request.symbol, emotion)
        return {"message": f"Updated {request.symbol} emotion to {request.emotion}"}
    except Exception as e:
        logger.error(f"Error in update_emotion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_strategy")
async def update_strategy(request: UpdateStrategyRequest):
    """Update trading strategy for a symbol"""
    try:
        try:
            strategy = TradingStrategy(request.strategy.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid strategy")
            
        intelligent_engine.update_symbol_strategy(request.symbol, strategy)
        return {"message": f"Updated {request.symbol} strategy to {request.strategy}"}
    except Exception as e:
        logger.error(f"Error in update_strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/toggle_symbol")
async def toggle_symbol(request: ToggleSymbolRequest):
    """Enable or disable trading for a symbol"""
    try:
        intelligent_engine.enable_symbol(request.symbol, request.enabled)
        status = "enabled" if request.enabled else "disabled"
        return {"message": f"Symbol {request.symbol} {status}"}
    except Exception as e:
        logger.error(f"Error in toggle_symbol: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available_strategies")
async def get_available_strategies():
    """Get list of available trading strategies"""
    return [strategy.value for strategy in TradingStrategy]

@app.get("/available_emotions")
async def get_available_emotions():
    """Get list of available emotion levels"""
    return [emotion.value for emotion in EmotionLevel]

@app.get("/symbol_configs")
async def get_symbol_configs():
    """Get all symbol configurations"""
    try:
        configs = {}
        for symbol, config in intelligent_engine.symbol_configs.items():
            configs[symbol] = {
                "strategy": config.strategy.value,
                "emotion": config.emotion.value,
                "max_positions": config.max_positions,
                "enabled": config.enabled,
                "custom_lot_size": config.custom_lot_size,
                "custom_timeframe": config.custom_timeframe,
                "daily_trades": config.daily_trades,
                "daily_pnl": config.daily_pnl
            }
        return configs
    except Exception as e:
        logger.error(f"Error in get_symbol_configs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance_report", response_model=PerformanceReport)
async def get_performance_report():
    """Get comprehensive performance report"""
    try:
        report = intelligent_engine.get_status_report()
        return report
    except Exception as e:
        logger.error(f"Error in get_performance_report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/adjust_emotions_based_on_performance")
async def adjust_emotions():
    """Adjust emotions for all symbols based on performance"""
    try:
        success = intelligent_engine.adjust_bot_emotions_based_on_performance()
        return {"message": "Emotions adjusted based on performance", "success": success}
    except Exception as e:
        logger.error(f"Error in adjust_emotions_based_on_performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_daily_stats")
async def reset_daily_stats():
    """Reset daily statistics for all symbols"""
    try:
        for symbol, config in intelligent_engine.symbol_configs.items():
            config.daily_trades = 0
            config.daily_pnl = 0.0
            intelligent_engine.performance_tracker[symbol]['trades_today'] = 0
            intelligent_engine.performance_tracker[symbol]['daily_pnl'] = 0.0
            intelligent_engine.performance_tracker[symbol]['wins'] = 0
            intelligent_engine.performance_tracker[symbol]['losses'] = 0
            intelligent_engine.performance_tracker[symbol]['win_rate'] = 0.0
        
        intelligent_engine.global_stats['daily_profit'] = 0.0
        intelligent_engine.global_stats['total_trades'] = 0
        
        return {"message": "Daily statistics reset"}
    except Exception as e:
        logger.error(f"Error in reset_daily_stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/setup_default_symbols")
async def setup_default_symbols():
    """Setup default symbols with different strategies"""
    try:
        success = intelligent_engine.setup_multi_symbol_trading()
        return {"message": "Default symbols setup", "success": success}
    except Exception as e:
        logger.error(f"Error in setup_default_symbols: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
