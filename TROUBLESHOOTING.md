# 🔧 Troubleshooting Guide

## Common Issues & Solutions

### Connection Problems

#### "Failed to connect to MT5"
**Symptoms**: Bot can't connect to broker
**Solutions**:
1. Check MT5 is running and logged in
2. Verify login credentials
3. Ensure internet connection is stable
4. Check if broker allows algorithmic trading
5. Restart MT5 terminal

#### "Invalid account info"
**Symptoms**: Login credentials rejected
**Solutions**:
1. Double-check username/password
2. Verify server name is correct
3. Ensure account is active (not suspended)
4. Check if account type supports MT5

### Trading Issues

#### "Invalid stops, Retcode: 10016"
**Symptoms**: Stop loss/take profit orders rejected
**Solutions**:
1. Check minimum stop distance for symbol
2. Ensure SL/TP are not too close to current price
3. Verify symbol is active and trading
4. Check broker-specific restrictions

#### "Failed to get tick data"
**Symptoms**: No price data for symbol
**Solutions**:
1. Ensure symbol is visible in MT5
2. Check if symbol is actively trading
3. Verify market hours for symbol
4. Try refreshing symbol list

#### "Insufficient funds"
**Symptoms**: Trade rejected due to balance
**Solutions**:
1. Check account balance
2. Verify margin requirements
3. Reduce lot size
4. Close other positions if needed

### Performance Issues

#### Bot not trading
**Symptoms**: Bot running but no trades
**Solutions**:
1. Check if market is open
2. Verify confidence thresholds
3. Check emotion level settings
4. Review strategy configuration
5. Check logs for error messages

#### Too many trades
**Symptoms**: Bot opening too many positions
**Solutions**:
1. Increase trade cooldown
2. Raise confidence thresholds
3. Reduce max daily trades
4. Check emotion level (greedy = more trades)

#### Bot closing positions too early
**Symptoms**: Profitable trades closed prematurely
**Solutions**:
1. Adjust profit zone thresholds
2. Change emotion level (less fearful)
3. Modify strategy settings
4. Check trailing stop configuration

### Technical Issues

#### "Module not found" errors
**Symptoms**: Python import errors
**Solutions**:
1. Activate virtual environment
2. Install requirements: `pip install -r requirements.txt`
3. Check Python version (3.8+ required)

#### Frontend not loading
**Symptoms**: Dashboard won't open
**Solutions**:
1. Check if backend is running
2. Verify frontend is started: `npm run dev`
3. Check browser console for errors
4. Ensure ports 8000 and 5173 are free

#### Database errors
**Symptoms**: Trade history not saving
**Solutions**:
1. Check file permissions
2. Ensure disk space available
3. Restart backend to recreate database
4. Check log files for specific errors

## Log Analysis

### Key Log Messages
- **INFO**: Normal operation
- **WARNING**: Potential issues
- **ERROR**: Problems that need attention
- **CRITICAL**: Serious issues requiring immediate action

### Where to Find Logs
- **Backend**: `Backend/New Version/smart_trading_bot.log`
- **Frontend**: Browser developer console
- **MT5**: Terminal logs

### Common Error Codes
- **10016**: Invalid stops
- **10004**: Market closed
- **10013**: Invalid request
- **10018**: Market closed
- **10019**: No trading

## Getting Help

### Self-Diagnosis
1. Check logs for error messages
2. Verify all services are running
3. Test with simple configuration
4. Check broker status

### When to Contact Support
- Persistent connection issues
- Unexplained trading behavior
- System crashes or freezes
- Data corruption

### Information to Provide
- Error messages from logs
- Steps to reproduce issue
- System configuration
- Broker information
