# 🛠️ Setup Guide

## Prerequisites

- **Python 3.8+** installed
- **Node.js 16+** installed
- **MetaTrader 5** terminal installed and configured
- **Broker account** with MT5 access

## Installation Steps

### 1. Clone/Download Project
```bash
# Navigate to your project directory
cd "path/to/your/project"
```

### 2. Backend Setup
```bash
# Navigate to backend
cd "Backend/New Version"

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install Python dependencies
pip install -r ../../requirements.txt
```

### 3. Frontend Setup
```bash
# Navigate to frontend
cd ../../frontend

# Install Node.js dependencies
npm install
```

### 4. MT5 Configuration
1. Open MetaTrader 5
2. Login to your broker account
3. Ensure symbols you want to trade are visible
4. Note your login credentials for the bot

## Running the Bot

### Start Backend
```bash
cd "Backend/New Version"
python main.py
```
Backend will start on: http://0.0.0.0:8000

### Start Frontend
```bash
cd frontend
npm run dev
```
Frontend will start on: http://localhost:5173

### First Time Setup
1. Open http://localhost:5173 in your browser
2. Click "Connect to MT5" and enter your credentials
3. Add trading symbols via the dashboard
4. Configure your preferred strategy and emotion level
5. Start the bot!

## Configuration Files

The bot uses several configuration files:
- `symbol_strategies.yaml` - Symbol-specific strategies
- `bot_state.py` - Bot state management
- Database files are created automatically

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.
