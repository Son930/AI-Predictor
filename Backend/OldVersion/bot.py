import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import torch
import joblib
import os
import logging
import time
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.preprocessing import MinMaxScaler

from AiModel import get_action
from feature_engineering import engineer_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths for saving model components
MODEL_DIR = "tft_model"


def load_prediction_model():
    """Load the TFT model and necessary components for prediction"""
    logger.info("Loading TFT model...")

    # Load the full model
    tft_model = torch.load(os.path.join(MODEL_DIR, "tft_full_model.pth"))

    # Load the training dataset for reference
    training = joblib.load(os.path.join(MODEL_DIR, "training_dataset.pkl"))

    # Load scaler
    scaler = joblib.load(os.path.join(MODEL_DIR, "close_scaler.pkl"))

    # Load parameters
    params = joblib.load(os.path.join(MODEL_DIR, "training_parameters.pkl"))

    return tft_model, training, scaler, params


def predict_next_price(df_features, tft_model, training, scaler, params):
    """Get prediction from TFT model"""
    # Validate input columns
    required_cols = ["group_id", "time_idx"] + params["time_varying_known_reals"]
    missing_cols = set(required_cols) - set(df_features.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Prepare data for TFT format
    df_copy = df_features.copy()

    # Add required columns for TFT
    df_copy["group_id"] = 0
    latest_time = df_copy.index.max()

    # Set time_idx (assuming most recent data point is the max time_idx from training + 1)
    max_time_idx = training.data["time_idx"].max()
    df_copy["time_idx"] = max_time_idx + 1

    # Ensure only necessary columns are present and in correct order
    df_copy = df_copy[required_cols]

    # Create a dataset for prediction
    predict_dataset = TimeSeriesDataSet(
        data=df_copy,
        time_idx="time_idx",
        target=params["target"],
        group_ids=params["group_ids"],
        max_encoder_length=params["max_encoder_length"],
        max_prediction_length=params["max_prediction_length"],
        time_varying_known_reals=params["time_varying_known_reals"],
        time_varying_unknown_reals=params["time_varying_unknown_reals"],
        target_normalizer=training.target_normalizer,
        scalers=training.scalers,
        categorical_encoders=training.categorical_encoders,
        allow_missing_timesteps=True,  # Handle gaps in time series
    )

    # Create dataloader
    predict_dataloader = predict_dataset.to_dataloader(train=False, batch_size=1)

    # Make prediction
    raw_predictions = tft_model.predict(predict_dataloader)
    prediction_value = raw_predictions.mean(1).item()

    # Inverse transform if needed
    if scaler is not None:
        prediction_value = scaler.inverse_transform([[prediction_value]])[0][0]

    return prediction_value


def fetch_latest_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H1, n_bars=1):
    """
    Fetch the latest data from MT5
    """
    if not mt5.initialize():
        logger.error("MT5 init failed: %s", mt5.last_error())
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None:
        logger.error("Failed to fetch data: %s", mt5.last_error())
        return None

    df = pd.DataFrame(rates, columns=["time", "open", "high", "low", "close", "tick_volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"time": "timestamp", "tick_volume": "volume"}, inplace=True)
    df.set_index("timestamp", inplace=True)
    return df


def execute_trade(action, symbol="XAUUSD", lot=0.1):
    """
    Execute a trade on MT5
    """
    if not mt5.initialize():
        logger.error("MT5 init failed: %s", mt5.last_error())
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error("Failed to get tick: %s", mt5.last_error())
        return

    if action == "buy":
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "deviation": 10,  # Max price deviation in points
            "magic": 123456,  # Unique order ID
            "comment": "TFT Bot Buy",
            "type_time": mt5.ORDER_TIME_GTC,  # Good till canceled
            "type_filling": mt5.ORDER_FILLING_IOC  # Immediate or cancel
        }
    elif action == "sell":
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_SELL,
            "price": tick.bid,
            "deviation": 10,
            "magic": 123456,
            "comment": "TFT Bot Sell",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }
    else:
        return  # No action for "hold"

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error("Order failed: %s, Error: %s", result.comment, mt5.last_error())
    else:
        logger.info("Order executed: %s, Ticket: %s", action, result.order)


def trading_bot(threshold=0.0005, sleep_interval=3600):
    """
    Main trading bot function using TFT model
    """
    # Load model components
    tft_model, training, scaler, params = load_prediction_model()
    logger.info("TFT trading bot started...")

    while True:
        try:
            latest_data = fetch_latest_data()
            if latest_data is None:
                logger.warning("Failed to fetch data, retrying in 60 seconds")
                time.sleep(60)  # Retry after 1 min if failed
                continue

            # Engineer features
            features = engineer_features(latest_data)

            # Get prediction
            pred_price = predict_next_price(features, tft_model, training, scaler, params)

            # Determine action
            action = get_action(features, pred_price, threshold)

            # Log information
            current_price = features["close"].values[0]
            logger.info(
                f"Time: {features.index[0]}, Current: {current_price:.2f}, Predicted: {pred_price:.2f}, Action: {action}")

            # Execute trade
            if action in ["buy", "sell"]:
                execute_trade(action)

            # Wait for next interval
            logger.info(f"Waiting {sleep_interval} seconds until next prediction")
            time.sleep(sleep_interval)

        except Exception as e:
            logger.error(f"Error in trading bot: {str(e)}")
            time.sleep(60)  # Wait a minute and try again


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the TFT trading bot.")
    parser.add_argument("--threshold", type=float, default=0.0005, help="Prediction threshold for trade execution")
    parser.add_argument("--sleep_interval", type=int, default=3600, help="Time interval between predictions (in seconds)")
    args = parser.parse_args()

    trading_bot(threshold=args.threshold, sleep_interval=args.sleep_interval)