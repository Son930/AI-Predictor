# 🔌 API Reference

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required (local use only)

## Endpoints

### Bot Control

#### Start Bot
```http
POST /start_bot
```
**Response**: Bot status and confirmation

#### Stop Bot
```http
POST /stop_bot
```
**Response**: Bot stopped confirmation

#### Pause Bot
```http
POST /pause_bot
```
**Response**: Bot paused confirmation

#### Get Bot Status
```http
GET /bot_status
```
**Response**:
```json
{
  "running": true,
  "paused": false,
  "connected": true,
  "error_count": 0,
  "last_error": null
}
```

### MT5 Connection

#### Connect to MT5
```http
POST /connect_mt5
```
**Body**:
```json
{
  "login": "your_login",
  "password": "your_password",
  "server": "your_server"
}
```

#### Disconnect from MT5
```http
POST /disconnect_mt5
```

#### Get Connection Status
```http
GET /mt5_status
```

### Trading Operations

#### Execute Trade
```http
POST /execute_trade
```
**Body**:
```json
{
  "symbol": "EURUSD",
  "action": "buy",
  "lot_size": 0.1
}
```

#### Get Open Positions
```http
GET /positions
```
**Response**: Array of open positions

#### Close Position
```http
POST /close_position
```
**Body**:
```json
{
  "ticket": 12345
}
```

#### Modify Position
```http
POST /modify_position
```
**Body**:
```json
{
  "ticket": 12345,
  "stop_loss": 1.0850,
  "take_profit": 1.0950
}
```

### Configuration

#### Get Symbols
```http
GET /symbols
```

#### Add Symbol
```http
POST /add_symbol
```
**Body**:
```json
{
  "symbol": "GBPUSD",
  "strategy": "balanced",
  "emotion": "standard"
}
```

#### Remove Symbol
```http
DELETE /remove_symbol/{symbol}
```

#### Get Strategies
```http
GET /strategies
```

#### Update Strategy
```http
PUT /update_strategy
```
**Body**:
```json
{
  "symbol": "EURUSD",
  "strategy": "aggressive",
  "emotion": "confident"
}
```

### Performance

#### Get Trade History
```http
GET /trade_history
```

#### Get Performance Stats
```http
GET /performance_stats
```

#### Get Logs
```http
GET /logs?limit=100
```

## Data Models

### Position
```json
{
  "ticket": 12345,
  "symbol": "EURUSD",
  "type": "buy",
  "volume": 0.1,
  "open_price": 1.0850,
  "current_price": 1.0870,
  "profit": 20.0,
  "profit_pct": 0.18,
  "stop_loss": 1.0800,
  "take_profit": 1.0900,
  "open_time": "2025-08-28T15:00:00Z"
}
```

### Symbol Config
```json
{
  "symbol": "EURUSD",
  "strategy": "balanced",
  "emotion": "standard",
  "lot_size": 0.1,
  "enabled": true
}
```

### Strategy Config
```json
{
  "confidence_threshold": 0.75,
  "max_positions_per_symbol": 2,
  "position_size_multiplier": 1.0,
  "stop_loss_multiplier": 1.0,
  "take_profit_multiplier": 2.0,
  "trade_cooldown_minutes": 5,
  "risk_per_trade_percent": 1.0,
  "trailing_stop_distance": 20
}
```

## Error Responses

### Standard Error Format
```json
{
  "error": "Error description",
  "details": "Additional information",
  "timestamp": "2025-08-28T15:00:00Z"
}
```

### Common HTTP Status Codes
- **200**: Success
- **400**: Bad Request
- **404**: Not Found
- **500**: Internal Server Error

## WebSocket Events

### Real-time Updates
- **position_update**: Position opened/closed/modified
- **price_update**: Symbol price changes
- **bot_status**: Bot state changes
- **error**: Error notifications

## Rate Limits
- No rate limiting currently implemented
- Consider adding for production use
```

```markdown:DEPLOYMENT.md
#  Deployment Guide

## Local Development

### Prerequisites
- Python 3.8+
- Node.js 16+
- MetaTrader 5
- Virtual environment (recommended)

### Development Setup
```bash
# Backend
cd "Backend/New Version"
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r ../../requirements.txt
python main.py

# Frontend
cd frontend
npm install
npm run dev
```

## Production Deployment

### Option 1: Local Machine (Recommended)

#### Advantages
- No hosting costs
- Full control
- Easy maintenance
- No internet dependency

#### Requirements
- Windows machine (for MT5)
- Stable internet connection
- 24/7 operation capability

#### Setup
1. Install Python and Node.js
2. Clone project to machine
3. Configure MT5 connection
4. Set up Windows Task Scheduler for auto-start
5. Configure firewall rules

### Option 2: VPS/Cloud Hosting

#### Advantages
- Remote access
- Professional setup
- Scalability
- Backup options

#### Challenges
- MT5 connection complexity
- Higher costs
- Technical complexity
- VPS providers may block trading

#### Recommended Providers
- **DigitalOcean**: $5-10/month
- **Linode**: $5-10/month
- **Vultr**: $5-10/month

#### Setup Steps
1. Choose VPS provider
2. Install Ubuntu/Debian
3. Install Python and dependencies
4. Set up MT5 connection (complex)
5. Configure firewall and security
6. Set up monitoring and backups

### Option 3: Windows VPS

#### Advantages
- Native MT5 support
- Easier setup
- Better compatibility

#### Providers
- **Contabo**: €4.99/month
- **OVH**: €3.50/month
- **Hetzner**: €4.15/month

## Packaging as Executable

### Using PyInstaller

#### Install PyInstaller
```bash
pip install pyinstaller
```

#### Create Executable
```bash
cd "Backend/New Version"
pyinstaller --onefile --add-data "models/*;models/" --add-data "*.yaml;." main.py
```

#### Package Frontend
```bash
cd frontend
npm run build
# Copy dist folder to backend executable location
```

#### Distribution
1. Create zip file with:
   - Backend executable
   - Frontend build folder
   - Configuration files
   - README and setup guide

### Using Auto-py-to-exe (GUI Alternative)
1. Install: `pip install auto-py-to-exe`
2. Run: `auto-py-to-exe`
3. Select main.py
4. Choose one-file option
5. Add data files
6. Build executable

## Configuration Management

### Environment Variables
```bash
# Create .env file
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
BOT_STRATEGY=balanced
BOT_EMOTION=standard
```

### Configuration Files
- **symbol_strategies.yaml**: Trading strategies
- **bot_state.py**: Bot configuration
- **config.json**: User preferences

## Security Considerations

### Local Deployment
- No external access needed
- Minimal security concerns
- Regular updates recommended

### Remote Deployment
- Use HTTPS
- Implement authentication
- Restrict access to trusted IPs
- Regular security updates
- Monitor access logs

## Monitoring & Maintenance

### Log Management
- Rotate logs regularly
- Monitor error rates
- Set up alerts for critical issues

### Performance Monitoring
- Track trading performance
- Monitor system resources
- Check MT5 connection stability

### Backup Strategy
- Regular configuration backups
- Database backups
- Log archives
- Disaster recovery plan

## Troubleshooting Deployment

### Common Issues
1. **Executable not working**
   - Check dependencies
   - Verify file paths
   - Test on clean machine

2. **MT5 connection fails**
   - Check credentials
   - Verify server status
   - Test manual connection

3. **Performance issues**
   - Monitor system resources
   - Check network latency
   - Optimize configuration

## Support & Distribution

### For Friends/Users
1. **Simple Setup Guide**
   - Download and extract
   - Run executable
   - Configure MT5 connection
   - Start trading

2. **Configuration Help**
   - Basic settings guide
   - Strategy explanations
   - Risk management tips

3. **Troubleshooting**
   - Common issues
   - Contact information
   - Update procedures
```

This gives you a complete documentation structure! The key files are:

1. **README.md** - Project overview and quick start
2. **SETUP.md** - Detailed installation steps
3. **USER_GUIDE.md** - How to actually use the bot
4. **TROUBLESHOOTING.md** - Fix common problems
5. **API_REFERENCE.md** - Technical backend details
6. **DEPLOYMENT.md** - How to package and distribute

Would you like me to:
1. **Create any specific sections** in more detail?
2. **Add screenshots or examples** to any guides?
3. **Focus on a particular aspect** like the .exe packaging?
4. **Create a simple user manual** for your friends?

What feels most important to tackle next?
