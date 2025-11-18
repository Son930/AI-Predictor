# 🤖 Smart Trading Bot

An intelligent, multi-strategy trading bot with machine learning predictions, emotion-based risk management, and automated profit-taking for MetaTrader 5.

## ✨ Features

- **🤖 AI-Powered Trading**: ML-based price prediction using XGBoost
- **📊 Multi-Strategy Support**: Scalping, Balanced, Aggressive, Swing trading
- **💭 Emotion-Based Risk Management**: Adapts to market conditions and user psychology
- **💰 Proactive Profit-Taking**: Automatic position closure at profit targets
- **⏰ Market Hours Automation**: Starts/stops trading during market hours
- **📱 Modern Dashboard**: React-based UI with real-time monitoring
- **🔒 Robust Error Handling**: Comprehensive logging and fallback mechanisms

## 🚀 Quick Start

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
TestingAI/
├── Backend/
│   ├── New Version/          # Current production version
│   │   ├── main.py           # FastAPI application entry point
│   │   ├── api_routes.py     # API endpoint definitions
│   │   ├── trading_logicv3.py # Core trading logic with ML predictions
│   │   ├── feature_engineering.py # Feature extraction for ML models
│   │   ├── mt5_utils.py      # MetaTrader 5 connection utilities
│   │   ├── bot_state.py      # Bot state management
│   │   ├── symbol_manager.py # Symbol and strategy management
│   │   ├── rl_brain.py       # Reinforcement learning components
│   │   ├── train_rl.py       # RL model training script
│   │   └── symbol_strategies.yaml # Strategy configurations
│   ├── OldVersion/           # Legacy bot versions
│   └── OldVersionV2/         # Previous iteration
├── frontend/
│   ├── src/
│   │   ├── Comp/             # React components
│   │   │   ├── BotDashboard.jsx    # Main trading dashboard
│   │   │   ├── PositionMonitor.jsx # Real-time position tracking
│   │   │   ├── MT5Login.jsx        # MT5 connection interface
│   │   │   └── Settings.jsx        # Configuration panel
│   │   ├── Pages/
│   │   │   └── Home.jsx      # Home page
│   │   └── assets/           # Fonts and static assets
│   └── package.json          # Frontend dependencies
├── requirements.txt          # Python dependencies
├── SETUP.md                  # Detailed setup instructions
├── API_REFERENCE.md          # Complete API documentation
├── USER_GUIDE.md            # User manual
├── DEPLOYMENT.md             # Deployment guide
└── TROUBLESHOOTING.md        # Common issues and solutions
```

## 🛠️ Technologies Used

### Backend
- **FastAPI** - Modern Python web framework
- **PyTorch** - Deep learning framework for ML models
- **XGBoost** - Gradient boosting for price predictions
- **Pandas & NumPy** - Data manipulation and analysis
- **MetaTrader 5 API** - Trading platform integration
- **APScheduler** - Task scheduling for automated trading
- **SQLite** - Trade history storage

### Frontend
- **React 19** - UI framework
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Chart.js** - Data visualization
- **Axios** - HTTP client
- **React Router** - Client-side routing

## ⚙️ Configuration

### Strategy Configuration
Edit `Backend/New Version/symbol_strategies.yaml` to customize trading strategies per symbol:

```yaml
EURUSD:
  strategy: "scalping"
  emotion_level: "balanced"
  risk_per_trade: 0.02
```

### Environment Variables
Create a `.env` file (optional) for configuration:
- MT5 credentials (or use UI login)
- API keys (if using external data sources)

## 📚 Documentation

- **[SETUP.md](SETUP.md)** - Installation and setup guide
- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API endpoint documentation
- **[USER_GUIDE.md](USER_GUIDE.md)** - User manual and features
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment instructions
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

## 🔧 Development

### Running in Development Mode

**Backend:**
```bash
cd "Backend/New Version"
python main.py
```

**Frontend:**
```bash
cd frontend
npm run dev
```

### Training ML Models
```bash
cd "Backend/New Version"
python train_rl.py
```

## 📝 Features in Detail

- **AI-Powered Predictions**: Uses XGBoost and PyTorch models to predict price movements
- **Multi-Strategy Trading**: Supports scalping, balanced, aggressive, and swing trading strategies
- **Emotion-Based Risk Management**: Adjusts risk based on market conditions and user preferences
- **Real-Time Monitoring**: Live dashboard with position tracking and performance metrics
- **Automated Profit-Taking**: Closes positions automatically when profit targets are reached
- **Market Hours Detection**: Automatically starts/stops trading during market hours

## ⚠️ Disclaimer

This trading bot is for educational and research purposes. Trading involves substantial risk of loss. Always test thoroughly in a demo environment before using real funds. The authors are not responsible for any financial losses.

## 📄 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please refer to [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open an issue on GitHub.