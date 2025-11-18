import pandas as pd
import torch
import joblib
import os
import logging
import numpy as np
import sys
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.trainer import Trainer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pytorch_lightning as pl
import argparse
from tqdm import tqdm

# Setup logging with more comprehensive format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tft_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Paths for saving model components
MODEL_DIR = "tft_model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Debug CUDA setup (run once at startup)
logger.info(f"PyTorch version: {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available! Ensure you have a GPU-enabled PyTorch installation.")

logger.info(f"CUDA version: {torch.version.cuda}")
logger.info(f"GPU device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(0)}")
torch.cuda.empty_cache()


# Wrapper for TemporalFusionTransformer with improved error handling
class TFTWrapper(pl.LightningModule):
    def __init__(self, tft_model, learning_rate=0.001):
        super().__init__()
        self.tft_model = tft_model
        self.learning_rate = learning_rate
        self.save_hyperparameters("learning_rate")

    def forward(self, x):
        return self.tft_model(x)

    def training_step(self, batch, batch_idx):
        try:
            x, y = batch
            if y is None or len(y) == 0:
                logger.warning(f"Batch {batch_idx} has None or empty target. Skipping.")
                return None

            y_hat = self.tft_model(x)
            loss = self.tft_model.loss(y_hat, y)
            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

            # Log memory usage occasionally but not every batch
            if torch.cuda.is_available() and batch_idx % 100 == 0:
                mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2
                mem_reserved = torch.cuda.memory_reserved() / 1024 ** 2
                logger.debug(f"GPU Memory: Allocated={mem_allocated:.2f}MB, Reserved={mem_reserved:.2f}MB")

            return loss
        except Exception as e:
            logger.error(f"Error in training_step: {str(e)}")
            raise e

    def validation_step(self, batch, batch_idx):
        try:
            x, y = batch
            if y is None or len(y) == 0:
                logger.warning(f"Val batch {batch_idx} has None or empty target. Skipping.")
                return None

            y_hat = self.tft_model(x)
            loss = self.tft_model.loss(y_hat, y)
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            return loss
        except Exception as e:
            logger.error(f"Error in validation_step: {str(e)}")
            raise e

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }


def load_and_prepare_data(data_path):
    """Separate function to load and prepare data to handle errors better"""
    logger.info("Loading data...")
    try:
        # Use tqdm to show progress during loading
        with tqdm(desc="Reading CSV chunks", unit="chunk") as pbar:
            chunks = []
            for chunk in pd.read_csv(data_path, chunksize=100000):
                chunks.append(chunk)
                pbar.update(1)
            df = pd.concat(chunks, ignore_index=True)

        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        # Convert timestamp and check for issues
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Check and handle missing data
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Dataset contains {missing_count} missing values. Filling with appropriate values.")
            df = df.fillna(0)

        # Check for infinite values
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            logger.warning(f"Dataset contains {inf_count} infinite values. Replacing them.")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)

        # Data summary
        logger.info("First few rows of the dataset:")
        logger.info(df.head().to_string())

        # Add required columns
        df["group_id"] = 0
        df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.days

        # Ensure the 'Target' column exists and is valid
        if 'Target' not in df.columns:
            raise ValueError("The 'Target' column is missing from the dataset.")

        return df

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def train_tft_model(data_path, dynamic_epochs=True, max_epochs=50, min_epochs=10, patience=5,
                    batch_size=128, learning_rate=0.001, hidden_size=128, encoder_length=60):
    """Main training function with better error handling and progress reporting"""
    try:
        # Load and prepare data (now as separate function)
        df = load_and_prepare_data(data_path)

        feature_columns = [col for col in df.columns if col not in
                           ["timestamp", "group_id", "time_idx", "Target", "RSI_category", "trade_signal"]]

        logger.info(f"Feature columns: {feature_columns}")

        # Scale the close price
        scaler = MinMaxScaler()
        df["close_scaled"] = scaler.fit_transform(df[["close"]])
        joblib.dump(scaler, os.path.join(MODEL_DIR, "close_scaler.pkl"))
        logger.info("Price scaling complete and scaler saved")

        target = "Target"
        time_varying_known_reals = feature_columns
        time_varying_unknown_reals = [target]

        # Create datasets with progress indicators
        logger.info("Creating training and validation datasets...")
        max_prediction_length = 1
        training_cutoff = int(len(df) * 0.8)

        training = TimeSeriesDataSet(
            data=df[:training_cutoff],
            time_idx="time_idx",
            target=target,
            group_ids=["group_id"],
            max_encoder_length=encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
            allow_missing_timesteps=True,
            add_relative_time_idx=True,  # Add as a feature
        )

        validation = TimeSeriesDataSet.from_dataset(
            training, df[training_cutoff:], predict=True, allow_missing_timesteps=True
        )
        logger.info("Datasets created successfully")

        # Create dataloaders with appropriate settings
        num_workers = 0 if os.name == 'nt' else 4  # Use workers on non-Windows

        train_dataloader = training.to_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
        )

        val_dataloader = validation.to_dataloader(
            train=False,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
        )

        logger.info("Checking first batch...")
        batch_iterator = iter(train_dataloader)
        try:
            first_batch = next(batch_iterator)
            x, y = first_batch
            # Print minimal info about shape, not full content
            logger.info(f"First batch x keys: {list(x.keys())}")
            logger.info(f"First batch y shape: {[yi.shape for yi in y]}")
            logger.info("First batch check completed successfully")

            # Free memory explicitly
            del batch_iterator, first_batch, x, y
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error checking first batch: {str(e)}")
            # Continue anyway

        # Set up callbacks with checkpoint saving and progress bar
        logger.info("Setting up training callbacks...")

        # Set up callbacks with checkpoint saving and progress bar
        callbacks = [
            LearningRateMonitor(),
            TQDMProgressBar(refresh_rate=10),  # Real-time progress bar
            ModelCheckpoint(
                dirpath=os.path.join(MODEL_DIR, "checkpoints"),
                filename="tft-{epoch:02d}-{val_loss:.4f}",
                save_top_k=3,
                monitor="val_loss",
                mode="min",
            ),
        ]

        if dynamic_epochs:
            callbacks.append(EarlyStopping(
                monitor="val_loss",
                patience=patience,
                mode="min",
                verbose=True
            ))

        # Calculate appropriate gradient accumulation
        effective_batch_size = 256
        accumulate_grad_batches = max(1, effective_batch_size // batch_size)

        logger.info(f"Setting up trainer with max_epochs={max_epochs}, "
                    f"accumulate_grad_batches={accumulate_grad_batches}")

        # Set up trainer with appropriate devices
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available!")

        accelerator = "gpu"
        precision = "16-mixed"

        pl_trainer = Trainer(
            max_epochs=max_epochs if dynamic_epochs else min_epochs,
            min_epochs=min_epochs,
            accelerator=accelerator,
            devices=1,
            gradient_clip_val=0.1,
            callbacks=callbacks,
            enable_progress_bar=True,
            precision=precision,
            accumulate_grad_batches=accumulate_grad_batches,
            logger=True,
            log_every_n_steps=10,
        )

        # Create the model with appropriate hyperparameters
        logger.info("Creating TFT model...")
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=4,
            dropout=0.2,  # Slightly higher dropout for regularization
            hidden_continuous_size=64,  # Increased from 32
            loss=QuantileLoss(),
        ).to("cuda")

        tft_wrapped = TFTWrapper(tft, learning_rate=learning_rate)

        # Train with proper exception handling
        logger.info("Starting model training...")
        try:
            pl_trainer.fit(
                tft_wrapped,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

        # Evaluate model
        logger.info("Evaluating model on validation set...")
        try:
            actuals_list = []
            preds_list = []

            # Use tqdm for progress during prediction
            for batch_idx, (x, y) in enumerate(tqdm(val_dataloader, desc="Evaluating")):
                if y is None or len(y) == 0:
                    logger.warning(f"Skipping evaluation batch {batch_idx} with None target")
                    continue

                # Get predictions
                predictions = tft.predict(x)
                mean_preds = predictions.mean(1)  # Mean over quantiles

                # Convert to numpy for easier handling
                mean_preds_np = mean_preds.detach().cpu().numpy()
                actuals_np = y[0].detach().cpu().numpy()

                preds_list.append(mean_preds_np)
                actuals_list.append(actuals_np)

            # Combine results
            all_preds = np.concatenate(preds_list)
            all_actuals = np.concatenate(actuals_list)

            # Calculate metrics
            rmse = mean_squared_error(all_actuals, all_preds, squared=False)
            mae = np.mean(np.abs(all_actuals - all_preds))

            logger.info(f"Evaluation metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

            # Save metrics for reference
            with open(os.path.join(MODEL_DIR, "metrics.txt"), "w") as f:
                f.write(f"RMSE: {rmse:.4f}\n")
                f.write(f"MAE: {mae:.4f}\n")

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")

        # Save the model
        logger.info("Saving final model...")
        try:
            torch.save(tft.state_dict(), os.path.join(MODEL_DIR, "tft_model.pth"))
            torch.save(tft, os.path.join(MODEL_DIR, "tft_full_model.pth"))
            joblib.dump(training, os.path.join(MODEL_DIR, "training_dataset.pkl"))

            # Save a sample batch for inference testing
            joblib.dump(first_batch, os.path.join(MODEL_DIR, "sample_batch.pkl"))

            # Save model hyperparameters
            with open(os.path.join(MODEL_DIR, "model_info.txt"), "w") as f:
                f.write(f"Encoder length: {encoder_length}\n")
                f.write(f"Hidden size: {hidden_size}\n")
                f.write(f"Learning rate: {learning_rate}\n")
                f.write(f"Feature columns: {', '.join(feature_columns)}\n")

            logger.info("Model training complete and all artifacts saved.")
            return tft, training, scaler

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Critical error in training process: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TFT model with enhanced feedback and performance.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the engineered data CSV file")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of training epochs")
    parser.add_argument("--min_epochs", type=int, default=10, help="Minimum number of training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size for TFT model")
    parser.add_argument("--encoder_length", type=int, default=60, help="Max encoder length (lookback period)")

    args = parser.parse_args()

    try:
        logger.info("Starting TFT model training process...")
        train_tft_model(
            data_path=args.data_path,
            max_epochs=args.max_epochs,
            min_epochs=args.min_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            encoder_length=args.encoder_length,
        )
        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"Training process failed: {str(e)}")
        sys.exit(1)