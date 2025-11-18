# rl_brain.py
import numpy as np
import torch
from feature_engineering import engineer_features

class RLBrain:
    """
    Minimal RL brain for trading:
    - Loads a trained PyTorch policy (or you can adapt for SB3 .zip model)
    - Takes engineered_df row as input, outputs action {0=hold, 1=long, 2=short, 3=close}
    """
    def __init__(self, model_path, window_size=64, device="cpu"):
        self.window_size = window_size
        self.device = device
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        self.feature_cols = None
        self.buffer = []

    def update_features(self, raw_df):
        """
        raw_df: recent OHLCV DataFrame (with datetime index)
        Updates rolling buffer for RL input.
        """
        df_eng = engineer_features(raw_df, fast=True, include_slow=False, drop_na=True)
        if self.feature_cols is None:
            self.feature_cols = [c for c in df_eng.columns if c != "Target"]
        latest_row = df_eng[self.feature_cols].iloc[-1].to_numpy(dtype=np.float32)
        self.buffer.append(latest_row)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def predict_action(self):
        """
        Returns RL action if buffer full, else hold (0).
        """
        if len(self.buffer) < self.window_size:
            return 0  # hold until buffer ready
        obs = np.expand_dims(np.array(self.buffer, dtype=np.float32), axis=0)
        with torch.no_grad():
            logits = self.model(torch.tensor(obs, device=self.device))
            action = int(torch.argmax(logits, dim=1).item())
        return action
