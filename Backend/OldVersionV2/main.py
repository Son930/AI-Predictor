# main.py
from datetime import datetime
from http.client import HTTPException
from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import asyncio
from config import *
from endpoints import router
from trading_logic import TradingStrategy
from mt5_core import MT5Manager
from ml_engine import ForexModel, FeatureEngineer
from schemas import TradeRequest
import uvicorn
import logging

logger = logging.getLogger(__name__)

# Global instances
mt5_manager = None
trading_strategy = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Manage application lifecycle."""
    global mt5_manager, trading_strategy
    
    # Startup
    try:
        logger.info("Initializing trading bot components")
        
        # Initialize MT5 connection (replace with your credentials)
        mt5_manager = MT5Manager(
            login=5034454017,  # Replace with your MT5 account number
            password="BuE!O6Nm",  # Replace with your MT5 password
            server="MetaQuotes-Demo"  # Replace with your MT5 server
        )
        if not mt5_manager.initialize():
            raise RuntimeError("Failed to initialize MT5 connection")
            
        # Initialize model and feature engineer
        feature_engineer = FeatureEngineer()
        forex_model = ForexModel(load_existing=True)
        
        # Initialize trading strategy
        trading_strategy = TradingStrategy()
        
        # Initial training
        logger.info("Performing initial model training")
        df = mt5_manager.fetch_historical_data(
            state.symbol,
            state.timeframe,
            state.num_bars
        )
        if df is None or df.empty:
            raise RuntimeError("Failed to fetch initial data for training")
        
        processed_df = feature_engineer.process(df)
        metrics, importances = forex_model.train(processed_df)
        if metrics is None:
            raise RuntimeError("Initial model training failed")
        
        logger.info(f"Initial model trained successfully - Metrics: {metrics}")
        state.last_data_refresh = datetime.now()
        
        # Update bot state
        state.running = True
        state.model = forex_model
        logger.info("Trading bot initialized successfully")
        
        # Start background task for automated trading
        asyncio.create_task(automated_trading_task())
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        state.record_error(str(e))
        raise
        
    yield  # Application runs here
    
    # Shutdown
    try:
        logger.info("Shutting down trading bot")
        state.running = False
        if mt5_manager:
            mt5_manager.close()
        state.save_state()
        logger.info("Shutdown completed")
    except Exception as e:
        logger.error(f"Shutdown failed: {str(e)}", exc_info=True)

async def automated_trading_task():
    """Background task to refresh data, train model, execute trades, and manage positions."""
    while state.running:
        try:
            if not state.paused:
                # Fetch fresh data
                df = mt5_manager.fetch_historical_data(
                    state.symbol,
                    state.timeframe,
                    state.num_bars
                )
                
                if df is not None and not df.empty:
                    # Process features
                    processed_df = trading_strategy.feature_engineer.process(df)
                    current_price = df['close'].iloc[-1]
                    
                    # Retrain model periodically (every hour)
                    if (datetime.now() - state.last_data_refresh).total_seconds() > 3600:
                        metrics, importances = trading_strategy.model.train(processed_df)
                        if metrics:
                            logger.info(f"Model retrained - Metrics: {metrics}")
                            state.last_data_refresh = datetime.now()
                            
                    # Generate signal
                    signal = trading_strategy.generate_signal()
                    state.update_metrics(
                        signal.action,
                        signal.confidence,
                        signal.features
                    )
                    
                    # Manage existing positions
                    positions = mt5_manager.get_positions(state.symbol)
                    account_balance = mt5_manager.account_info().balance if mt5_manager.account_info() else 10000
                    total_profit = sum(pos.profit for pos in positions)
                    
                    for pos in positions:
                        # Calculate profit/loss percentage
                        entry_price = pos.price_open
                        profit_percent = ((current_price - entry_price) / entry_price * 100) if pos.type == mt5_manager.ORDER_TYPE_BUY else ((entry_price - current_price) / entry_price * 100)
                        
                        # Close losing positions (e.g., -0.5% loss)
                        if profit_percent < -0.5:
                            close_request = TradeRequest(
                                symbol=pos.symbol,
                                action="sell" if pos.type == mt5_manager.ORDER_TYPE_BUY else "buy",
                                lot_size=pos.volume
                            )
                            result = mt5_manager.execute_order(close_request)
                            if result["success"]:
                                logger.info(f"Closed losing position {pos.ticket}: {profit_percent:.2f}%")
                            else:
                                logger.error(f"Failed to close position {pos.ticket}: {result['message']}")
                                state.record_error(result["message"])
                        
                        # Take profit (e.g., +1% gain)
                        elif profit_percent > 1.0:
                            close_request = TradeRequest(
                                symbol=pos.symbol,
                                action="sell" if pos.type == mt5_manager.ORDER_TYPE_BUY else "buy",
                                lot_size=pos.volume
                            )
                            result = mt5_manager.execute_order(close_request)
                            if result["success"]:
                                logger.info(f"Closed profitable position {pos.ticket}: {profit_percent:.2f}%")
                            else:
                                logger.error(f"Failed to close position {pos.ticket}: {result['message']}")
                                state.record_error(result["message"])
                    
                    # Execute new trade if signal is buy or sell
                    if signal.action in ["buy", "sell"] and signal.confidence >= state.min_confidence:
                        volatility = processed_df['volatility'].iloc[-1] if 'volatility' in processed_df else 0.01
                        lot_size = trading_strategy.calculate_position_size(account_balance, volatility)
                        
                        trade_request = TradeRequest(
                            symbol=state.symbol,
                            action=signal.action,
                            lot_size=lot_size
                        )
                        result = mt5_manager.execute_order(trade_request)
                        if result["success"]:
                            logger.info(f"Automated trade executed: {signal.action} {lot_size} lots - Order: {result['message']}")
                        else:
                            logger.error(f"Trade execution failed: {result['message']}")
                            state.record_error(result["message"])
                    
                    # Update positions in state
                    positions = mt5_manager.get_positions(state.symbol)  # Refresh after potential closures
                    state.clear_positions()
                    for pos in positions:
                        state.update_position(pos.__dict__)
                    
                    # Check risk limits
                    total_profit = sum(pos.profit for pos in positions)
                    if account_balance > 0 and total_profit < -(account_balance * state.max_loss_percent / 100):
                        logger.warning("Max loss limit reached, pausing bot")
                        state.paused = True
                    
        except Exception as e:
            logger.error(f"Automated trading task error: {str(e)}", exc_info=True)
            state.record_error(str(e))
            if state.error_count > 5:
                logger.critical("Too many errors, shutting down bot")
                state.running = False
            
        await asyncio.sleep(60)  # Refresh every minute

# Initialize FastAPI app
app = FastAPI(title="AI Trading Bot", lifespan=lifespan)
app.include_router(router, prefix="/api/v1")

# Dependency to access trading strategy
def get_trading_strategy() -> TradingStrategy:
    if trading_strategy is None:
        raise HTTPException(status_code=503, detail="Trading Strategy not initialized")
    return trading_strategy

# Example endpoint
@app.get("/status")
async def get_status():
    return {
        "running": state.running,
        "paused": state.paused,
        "last_prediction": state.last_prediction,
        "last_confidence": state.last_confidence,
        "position_count": len(state.positions),
        "error_count": state.error_count
    }

if __name__ == "__main__":
    # Configure logging before anything else
    configure_logging(log_file="trading_bot.log", log_level=logging.INFO)
    
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for production
        log_level="info"
    )