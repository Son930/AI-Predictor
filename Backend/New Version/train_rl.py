import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from gym import spaces
from mt5_utils import fetch_historical_data
from feature_engineering import engineer_features
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Trading environment
class TradingEnv(gym.Env):
    def __init__(self, df, feature_cols, window_size=64):
        super(TradingEnv, self).__init__()
        self.df = df
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, len(feature_cols)), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 0=hold, 1=long, 2=short, 3=close
        self.current_step = window_size
        self.position = None  # None, 'long', or 'short'
        self.entry_price = None
        self.max_steps = len(df) - window_size - 1

    def reset(self):
        self.current_step = self.window_size
        self.position = None
        self.entry_price = None
        return self._get_observation()

    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        obs = self.df[self.feature_cols].iloc[start:end].to_numpy(dtype=np.float32)
        return obs

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        reward = 0.0
        current_price = self.df['close'].iloc[self.current_step]

        if action == 1 and self.position is None:  # Open long
            self.position = 'long'
            self.entry_price = current_price
        elif action == 2 and self.position is None:  # Open short
            self.position = 'short'
            self.entry_price = current_price
        elif action == 3 and self.position is not None:  # Close position
            if self.position == 'long':
                reward = (current_price - self.entry_price) / self.entry_price * 100
            elif self.position == 'short':
                reward = (self.entry_price - current_price) / self.entry_price * 100
            self.position = None
            self.entry_price = None
        # Penalize holding without position
        if action == 0 and self.position is None:
            reward = -0.01

        obs = self._get_observation()
        info = {}
        return obs, reward, done, info

# Policy network matching RLBrain expectations
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=4):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:  # Add batch dimension if missing
            x = x.unsqueeze(0)
        if x.dim() == 3:  # Flatten (batch_size, window_size, num_features)
            x = x.contiguous().reshape(x.size(0), -1)  # Use reshape instead of view
        return self.net(x)

# Train RL model for one symbol
def train_rl_for_symbol(symbol, timeframe=16385, num_bars=3500, episodes=1000, window_size=64):
    logger.info(f"Training RL model for {symbol}")
    # Fetch data
    df = fetch_historical_data(symbol, timeframe, num_bars)
    if df is None or len(df) < window_size + 1:
        logger.error(f"Failed to fetch sufficient data for {symbol}")
        return False

    df_eng = engineer_features(df, fast=True, include_slow=False, drop_na=True)
    feature_cols = [c for c in df_eng.columns if c != "Target" and c != "Future_Return"]
    if not feature_cols:
        logger.error(f"No valid features for {symbol}")
        return False

    # Initialize environment and model
    env = TradingEnv(df_eng, feature_cols, window_size)
    input_dim = window_size * len(feature_cols)
    policy = PolicyNetwork(input_dim).to('cpu')
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    gamma = 0.99

    # Training loop (simple policy gradient)
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to('cpu')
            logits = policy(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            m = Categorical(probs)
            action = m.sample()
            next_state, reward, done, _ = env.step(action.item())
            episode_reward += reward

            # Policy gradient update
            log_prob = m.log_prob(action)
            loss = -log_prob * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        if episode % 100 == 0:
            logger.info(f"{symbol} Episode {episode}/{episodes}, Reward: {episode_reward:.2f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f"models/{symbol}_rl_policy.pth"
    torch.save(policy, model_path)
    logger.info(f"Saved RL model for {symbol} to {model_path}")
    return True

# Train for all symbols
def train_all_symbols():
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    for symbol in symbols:
        success = train_rl_for_symbol(symbol)
        if not success:
            logger.warning(f"Failed to train RL model for {symbol}")

if __name__ == "__main__":
    import MetaTrader5 as mt5
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        exit()
    train_all_symbols()