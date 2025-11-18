import logging
import MetaTrader5 as mt5

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    filename="smart_trading_bot.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class BotState:
    def __init__(self):
        self.running = False
        self.paused = False
        self.model = None
        self.last_prediction = 0
        self.last_confidence = 0.0
        self.last_features = {}
        self.positions = []
        self.symbol = "EURUSD"
        self.timeframe = mt5.TIMEFRAME_H1
        self.num_bars = 3500
        self.lot_size = 0.1
        self.suggested_lot_size = 0.1
        self.error_count = 0
        self.last_error = None
        self.max_loss_percent = 1.2
        self.position_evaluation_interval = 200
        self.last_position_refresh = 0
        self.position_cache = []
        self.volatility = 0
        self.max_positions = 5
        self.max_total_risk_percent = 1
        self.mt5_connected = False
        self.mt5_login = None
        self.mt5_password = None
        self.mt5_server = None
        self.data_cache = None
        self.last_data_fetch = 0

state = BotState()