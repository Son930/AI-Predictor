import pandas as pd
import torch
import joblib
import os
import logging
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.trainer import Trainer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pytorch_lightning as pl
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths for saving model components
MODEL_DIR = "tft_model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Debug CUDA setup (run once at startup)
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"GPU device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA not available. Falling back to CPU.")

# Wrapper for TemporalFusionTransformer
class TFTWrapper(pl.LightningModule):
    def __init__(self, tft_model):
        super().__init__()
        self.tft_model = tft_model
        if torch.cuda.is_available():
            self.tft_model.cuda()

    def forward(self, x):
        return self.tft_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logger.info(f"Batch {batch_idx} - y: {y}")
        if y is None:
            logger.warning(f"Batch {batch_idx} has None target. Skipping.")
            return None
        if torch.cuda.is_available():
            x = {k: v.cuda() for k, v in x.items()}
            y = [yi.cuda() for yi in y]  # Handle list of tensors
        y_hat = self.tft_model(x)
        loss = self.tft_model.loss(y_hat, y)
        self.log("train_loss", loss)
        if torch.cuda.is_available() and batch_idx % 10 == 0:
            logger.debug(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logger.info(f"Batch {batch_idx} - y: {y}")
        if y is None:
            logger.warning(f"Batch {batch_idx} has None target. Skipping.")
            return None
        if torch.cuda.is_available():
            x = {k: v.cuda() for k, v in x.items()}
            y = [yi.cuda() for yi in y]  # Handle list of tensors
        y_hat = self.tft_model(x)
        loss = self.tft_model.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.tft_model.configure_optimizers()

def train_tft_model(data_path, dynamic_epochs=True, max_epochs=50, min_epochs=10, patience=5):
    logger.info("Loading and preparing data...")
    chunks = pd.read_csv(data_path, chunksize=100000)
    df = pd.concat(chunks, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Ensure the 'Target' column exists and is valid
    if 'Target' not in df.columns:
        raise ValueError("The 'Target' column is missing from the dataset.")
    if df['Target'].isnull().any():
        logger.warning("The 'Target' column contains NaN values. Filling with 0 or dropping.")
        df['Target'].fillna(0, inplace=True)  # or use df.dropna(subset=['Target'], inplace=True)

    # Check for NaN or inf values
    if df.isnull().any().any():
        logger.warning("Dataset contains NaN values. Filling or dropping them.")
        df.fillna(0, inplace=True)  # or use df.dropna(inplace=True)

    if df.isin([np.inf, -np.inf]).any().any():
        logger.warning("Dataset contains inf values. Replacing them.")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)  # or use df.dropna(inplace=True)

    df["group_id"] = 0
    df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.days

    feature_columns = [col for col in df.columns if col not in
                       ["timestamp", "group_id", "time_idx", "Target", "RSI_category", "trade_signal"]]

    scaler = MinMaxScaler()
    df["close_scaled"] = scaler.fit_transform(df[["close"]])
    joblib.dump(scaler, os.path.join(MODEL_DIR, "close_scaler.pkl"))

    target = "Target"
    time_varying_known_reals = feature_columns
    time_varying_unknown_reals = [target]

    max_encoder_length = 60
    max_prediction_length = 1
    training_cutoff = int(len(df) * 0.8)

    training = TimeSeriesDataSet(
        data=df[:training_cutoff],
        time_idx="time_idx",
        target=target,
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, df[training_cutoff:], predict=True, allow_missing_timesteps=True)

    batch_size = 128
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=0,  # Disable multiprocessing
        pin_memory=True,
        persistent_workers=False
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=0,  # Disable multiprocessing
        pin_memory=True,
        persistent_workers=False
    )

    # Check the first few rows of the dataset
    logger.info("First few rows of the dataset:")
    logger.info(df.head())

    # Check the first batch in the dataloader
    for batch in train_dataloader:
        x, y = batch
        logger.info(f"First training batch - x: {x}, y: {y}")
        break

    callbacks = [
        LearningRateMonitor(),
    ]
    if dynamic_epochs:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=patience, mode="min"))

    logger.info(f"Setting up trainer with max_epochs={max_epochs if dynamic_epochs else min_epochs}")
    pl_trainer = Trainer(
        max_epochs=max_epochs if dynamic_epochs else min_epochs,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=callbacks,
        enable_progress_bar=True,  # Built-in progress bar from PyTorch Lightning
        precision="16-mixed",  # Updated to mixed precision
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        loss=QuantileLoss(),
    )

    tft_wrapped = TFTWrapper(tft)

    logger.info("Training model...")
    pl_trainer.fit(
        tft_wrapped,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    logger.info("Evaluating model...")
    predictions = tft.predict(val_dataloader)
    actuals = torch.cat([y for x, y in iter(val_dataloader)])
    mean_preds = predictions.mean(1)  # Assuming you want the mean over the quantiles

    mean_preds_np = mean_preds.detach().cpu().numpy()
    actuals_np = actuals[:, 0].detach().cpu().numpy()

    rmse = mean_squared_error(actuals_np, mean_preds_np, squared=False)
    logger.info(f"RMSE: {rmse}")

    logger.info("Saving model for trading bot...")
    torch.save(tft.state_dict(), os.path.join(MODEL_DIR, "tft_model.pth"))
    torch.save(tft, os.path.join(MODEL_DIR, "tft_full_model.pth"))
    joblib.dump(training, os.path.join(MODEL_DIR, "training_dataset.pkl"))

    logger.info("Model training complete and saved.")
    return tft, training, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TFT model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the engineered data CSV file")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of training epochs")
    parser.add_argument("--min_epochs", type=int, default=10, help="Minimum number of training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    args = parser.parse_args()

    train_tft_model(
        data_path=args.data_path,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
    )