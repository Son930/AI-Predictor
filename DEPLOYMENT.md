# 🤖 Smart Trading Bot

An intelligent, multi-strategy trading bot with machine learning predictions, emotion-based risk management, and automated profit-taking for MetaTrader 5.

## ✨ Features

- ** AI-Powered Trading**: ML-based price prediction using XGBoost
- **📊 Multi-Strategy Support**: Scalping, Balanced, Aggressive, Swing trading
- ** Emotion-Based Risk Management**: Adapts to market conditions and user psychology
- ** Proactive Profit-Taking**: Automatic position closure at profit targets
- **⏰ Market Hours Automation**: Starts/stops trading during market hours
- **📱 Modern Dashboard**: React-based UI with real-time monitoring
- **🔒 Robust Error Handling**: Comprehensive logging and fallback mechanisms

##  Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   cd frontend && npm install
   ```

2. **Run Backend**
   ```bash
   cd "Backend/New Version"
   python main.py
   ```

3. **Run Frontend**
   ```bash
   cd frontend
   npm run dev
   ```

4. **Access Dashboard**: http://localhost:5173

## 📁 Project Structure

```
├── Backend/New Version/     # Main trading bot (FastAPI)
├── frontend/               # React dashboard
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

- **Broker Setup**: Configure MT5 connection in dashboard
- **Symbols**: Add trading pairs via dashboard or config
- **Risk Management**: Adjust emotion levels and strategy parameters
- **Market Hours**: Customize trading schedule

##  Documentation

- [Setup Guide](SETUP.md) - Detailed installation
- [User Guide](USER_GUIDE.md) - How to use the bot
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
- [API Reference](API_REFERENCE.md) - Backend endpoints

## ⚠️ Disclaimer

This is a trading bot for educational and personal use. Trading involves risk. Use at your own discretion.

